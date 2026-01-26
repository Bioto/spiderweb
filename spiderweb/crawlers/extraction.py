"""LLM-powered structured extraction from crawled web content.

Provides intelligent extraction using semantic guidance, custom queries,
and Pydantic schema validation with optional auto-improvement loops.
"""

import json
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel, ValidationError

from spiderweb.models.config import CrawlExtractionConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class CrawlExtractor:
    """LLM-powered structured extraction from crawled content.
    
    Uses LLM to extract structured data from web content with optional
    schema validation and iterative improvement.
    
    Example:
        >>> from pydantic import BaseModel
        >>> 
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: float
        ...     
        >>> extractor = CrawlExtractor(llm_client, config)
        >>> product = await extractor.extract(
        ...     html_content,
        ...     schema=Product,
        ...     semantic_guide="Extract product information"
        ... )
        >>> print(product.name, product.price)
    """
    
    def __init__(self, llm_client: Any, config: CrawlExtractionConfig | None = None):
        """Initialize crawl extractor.
        
        Args:
            llm_client: GlueLLM client for LLM operations
            config: Optional extraction configuration
        """
        self.llm_client = llm_client
        self.config = config or CrawlExtractionConfig()
        logger.debug("Initialized CrawlExtractor")
    
    def _get_schema_description(self, schema: type[BaseModel]) -> str:
        """Generate a description of the Pydantic schema for the LLM.
        
        Args:
            schema: Pydantic model class
            
        Returns:
            Human-readable schema description
        """
        schema_json = schema.model_json_schema()
        
        # Build description
        desc_parts = [f"Extract data matching this structure:"]
        desc_parts.append(f"Type: {schema.__name__}")
        
        if "description" in schema_json:
            desc_parts.append(f"Description: {schema_json['description']}")
        
        # Add field descriptions
        if "properties" in schema_json:
            desc_parts.append("\nFields:")
            for field_name, field_info in schema_json["properties"].items():
                field_type = field_info.get("type", "any")
                field_desc = field_info.get("description", "")
                required = field_name in schema_json.get("required", [])
                req_marker = " (required)" if required else " (optional)"
                desc_parts.append(f"  - {field_name}: {field_type}{req_marker}")
                if field_desc:
                    desc_parts.append(f"    {field_desc}")
        
        return "\n".join(desc_parts)
    
    def _build_extraction_prompt(
        self,
        content: str,
        schema: type[BaseModel] | None = None,
        semantic_guide: str | None = None,
        extraction_query: str | None = None,
    ) -> str:
        """Build the extraction prompt for the LLM.
        
        Args:
            content: Web content to extract from
            schema: Optional Pydantic schema
            semantic_guide: Optional semantic guidance
            extraction_query: Optional specific query
            
        Returns:
            Complete extraction prompt
        """
        prompt_parts = [
            "You are an expert at extracting structured information from web content.",
            ""
        ]
        
        # Add semantic guidance
        if semantic_guide:
            prompt_parts.append(f"Task: {semantic_guide}")
            prompt_parts.append("")
        
        # Add schema information
        if schema:
            prompt_parts.append(self._get_schema_description(schema))
            prompt_parts.append("")
        
        # Add specific query
        if extraction_query:
            prompt_parts.append(f"Specific instruction: {extraction_query}")
            prompt_parts.append("")
        
        # Add content
        prompt_parts.extend([
            "Content to extract from:",
            "---",
            content[:10000],  # Limit content length to avoid token issues
            "---",
            "",
        ])
        
        # Add output format instruction
        if schema:
            prompt_parts.append(
                "Provide the extracted data as valid JSON matching the schema above. "
                "Only return the JSON object, no additional text or markdown formatting."
            )
        else:
            prompt_parts.append(
                "Extract and return the relevant information as structured JSON. "
                "Only return the JSON object, no additional text or markdown formatting."
            )
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response: str, schema: type[T] | None = None) -> T | dict:
        """Parse LLM response and validate against schema.
        
        Args:
            response: LLM response text
            schema: Optional Pydantic schema for validation
            
        Returns:
            Validated Pydantic model instance or dict
            
        Raises:
            ValueError: If parsing or validation fails
        """
        # Clean response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            # Remove opening ```json or ```
            lines = response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)
        
        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        
        # Validate against schema if provided
        if schema:
            try:
                return schema(**data)
            except ValidationError as e:
                raise ValueError(f"LLM output does not match schema: {e}")
        
        return data
    
    async def extract(
        self,
        content: str,
        schema: type[T] | None = None,
        semantic_guide: str | None = None,
        extraction_query: str | None = None,
    ) -> T | dict | str:
        """Extract structured data from crawled content.
        
        Args:
            content: Web content (HTML or markdown)
            schema: Optional Pydantic model for output structure
            semantic_guide: Optional high-level extraction guidance
            extraction_query: Optional specific extraction instruction
            
        Returns:
            Extracted data as Pydantic model, dict, or string
            
        Raises:
            ValueError: If extraction or validation fails
        """
        # Use config values as defaults
        if semantic_guide is None:
            semantic_guide = self.config.semantic_guide
        if extraction_query is None:
            extraction_query = self.config.extraction_query
        if schema is None:
            schema = self.config.output_schema
        
        # Build prompt
        prompt = self._build_extraction_prompt(
            content,
            schema=schema,
            semantic_guide=semantic_guide,
            extraction_query=extraction_query,
        )
        
        logger.debug(f"Extracting with LLM (schema={schema.__name__ if schema else 'None'})")
        
        # Call LLM
        try:
            response = await self.llm_client.generate(
                prompt,
                temperature=self.config.temperature,
            )
            
            # Parse and validate response
            result = self._parse_llm_response(response.text, schema=schema)
            
            logger.info(f"Successfully extracted data (type={type(result).__name__})")
            
            return result
        
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            raise ValueError(f"Failed to extract structured data: {e}")
    
    async def extract_with_improvement(
        self,
        content: str,
        schema: type[T],
        semantic_guide: str | None = None,
        extraction_query: str | None = None,
    ) -> tuple[T, list[str]]:
        """Extract with iterative improvement loop.
        
        Attempts extraction, validates result, and if validation fails or
        quality is low, refines the prompt and tries again.
        
        Args:
            content: Web content to extract from
            schema: Pydantic model for output structure
            semantic_guide: Optional semantic guidance
            extraction_query: Optional specific query
            
        Returns:
            Tuple of (validated result, improvement log)
            
        Raises:
            ValueError: If extraction fails after all iterations
        """
        improvement_log: list[str] = []
        last_error: Exception | None = None
        
        for iteration in range(self.config.max_improve_iterations):
            try:
                logger.debug(f"Extraction attempt {iteration + 1}/{self.config.max_improve_iterations}")
                
                # Attempt extraction
                result = await self.extract(
                    content,
                    schema=schema,
                    semantic_guide=semantic_guide,
                    extraction_query=extraction_query,
                )
                
                # If we got here, extraction succeeded
                improvement_log.append(f"Iteration {iteration + 1}: Success")
                return result, improvement_log
            
            except ValueError as e:
                last_error = e
                improvement_log.append(f"Iteration {iteration + 1}: {e}")
                
                # If not the last iteration, refine the prompt
                if iteration < self.config.max_improve_iterations - 1:
                    # Add error feedback to extraction query
                    error_feedback = f"Previous attempt failed: {e}. Please correct the issues and ensure the output matches the schema exactly."
                    
                    if extraction_query:
                        extraction_query = f"{extraction_query}\n\n{error_feedback}"
                    else:
                        extraction_query = error_feedback
                    
                    logger.debug(f"Refining extraction prompt after error: {e}")
        
        # All iterations failed
        logger.error(f"Extraction failed after {self.config.max_improve_iterations} attempts")
        raise ValueError(f"Extraction failed after all improvement attempts: {last_error}")
    
    async def extract_batch(
        self,
        contents: list[str],
        schema: type[T] | None = None,
        semantic_guide: str | None = None,
        extraction_query: str | None = None,
    ) -> list[T | dict | str]:
        """Extract from multiple content pieces concurrently.
        
        Args:
            contents: List of web content to extract from
            schema: Optional Pydantic model for output structure
            semantic_guide: Optional semantic guidance
            extraction_query: Optional specific query
            
        Returns:
            List of extracted results
        """
        import asyncio
        
        tasks = [
            self.extract(
                content,
                schema=schema,
                semantic_guide=semantic_guide,
                extraction_query=extraction_query,
            )
            for content in contents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors but return successful results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch extraction failed for item {i}: {result}")
        
        return results


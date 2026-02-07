# Rich Live UI for Research Crawls

You can use Rich's **Live** display with **Layout** and **Progress** to show crawl4ai log output in a dedicated panel and progress bars per query.

## Concepts

1. **`Live`** – Refreshes a renderable in place. Use `screen=True` for a full alternate screen so the UI doesn’t scroll away.
2. **`Layout`** – Splits the terminal into regions (e.g. top = status + progress, bottom = log).
3. **`Panel`** – Puts content in a box; use one for “Crawl log” with the last N lines.
4. **`Progress`** – Multiple tasks (one per query); update as each query’s crawl advances/completes.
5. **Capturing crawl4ai output** – Crawl4ai prints `[INIT]`, `[FETCH]`, etc. to stdout. To show them in a panel you must capture stdout (e.g. redirect to a thread-safe buffer or use a custom logging handler) and render the last N lines in the layout.

## Minimal pattern

```python
import asyncio
import sys
from io import StringIO
from collections import deque
from threading import Lock

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
from rich.table import Table

console = Console()

# Thread-safe log buffer (last 20 lines) for the log panel
LOG_MAX_LINES = 20
log_lines: deque = deque(maxlen=LOG_MAX_LINES)
log_lock = Lock()

class LogCapture:
    """Redirect writes to a buffer for the Live log panel."""
    def __init__(self, original):
        self.original = original
    def write(self, s):
        if s.strip():
            with log_lock:
                for line in s.rstrip().split("\n"):
                    if line.strip():
                        log_lines.append(line)
        return self.original.write(s)
    def flush(self):
        return self.original.flush()

def make_layout(progress_table: Table, progress: Progress, log_content: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=10),
        Layout(name="log", ratio=1),
    )
    layout["top"].split_row(
        Layout(progress_table, name="table"),
        Layout(progress, name="progress"),
    )
    layout["log"].update(Panel(log_content, title="Crawl log", border_style="dim"))
    return layout

async def main():
    # Optional: capture stdout so crawl4ai output goes into the log panel
    # old_stdout = sys.stdout
    # sys.stdout = LogCapture(old_stdout)
    # try:
    #     ... run crawls ...
    # finally:
    #     sys.stdout = old_stdout

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        expand=True,
    )
    table = Table()
    table.add_column("Query")
    table.add_column("Status")
    table.add_column("URLs")

    def get_renderable():
        with log_lock:
            log_content = "\n".join(log_lines) if log_lines else "(no log yet)"
        return make_layout(table, progress, log_content)

    with progress:
        with Live(get_renderable(), refresh_per_second=4, screen=True) as live:
            task_ids = {}
            for i, query in enumerate(["query A", "query B"]):
                table.add_row(query, "Pending", "0")
                task_ids[query] = progress.add_task(f" {query[:40]}...", total=100)
            # Simulate updates
            for q in task_ids:
                for _ in range(10):
                    await asyncio.sleep(0.2)
                    progress.advance(task_ids[q], 10)
                    live.update(get_renderable())

if __name__ == "__main__":
    asyncio.run(main())
```

## Integrating with the research command

1. **Optional CLI flag** – e.g. `--live-ui` to enable the Live layout; otherwise keep current behavior.
2. **Capture stdout during Step 2** – Wrap the parallel crawl phase in a context that redirects `sys.stdout` to a `LogCapture` (or attach a logging handler that appends to `log_lines`). Restore stdout when leaving the context.
3. **Single Live context for Step 2** – Enter `Live(layout, refresh_per_second=4, screen=True)` at the start of the crawl loop. Build the layout from:
   - **Top row:** `create_progress_table()` (or a Table you update) + a `Progress` instance with one task per query.
   - **Bottom:** `Panel("\n".join(log_lines), title="Crawl log")`.
4. **Update layout on each refresh** – Either call `live.update(make_layout(...))` from a background refresh task or whenever you update `crawl_status` / progress. Use a lock around `log_lines` and the status dict so updates are thread-safe.
5. **Crawl4ai verbosity** – Crawl4ai’s `[INIT]` / `[FETCH]` etc. come from its own code. If it uses a `verbose` flag (e.g. on `BrowserConfig`), you can pass `verbose=False` to reduce noise when you don’t want a log panel, or leave it on when using the Live UI so the log panel has content. Our `Crawl4AICrawler` currently creates `AsyncWebCrawler()` with no config; you’d need to add a way to pass a crawl4ai config (e.g. `BrowserConfig(verbose=...)`) if you want to toggle that.

## References

- [Rich Live](https://rich.readthedocs.io/en/stable/live.html)
- [Rich Layout](https://rich.readthedocs.io/en/stable/layout.html)
- [Rich Progress](https://rich.readthedocs.io/en/stable/progress.html)
- Rich examples: `fullscreen.py`, `live_progress.py` (multiple progress in Live)

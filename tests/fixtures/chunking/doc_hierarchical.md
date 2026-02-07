# API Reference

This document describes the public API.

## Authentication

All requests require an API key in the `Authorization` header.

### Obtaining a key

Sign up at the dashboard and create a key in Settings.

## Endpoints

### GET /users

Returns a list of users.

#### Parameters

- `limit` (optional): Maximum number of results.
- `offset` (optional): Pagination offset.

### POST /users

Creates a new user.

#### Request body

- `name` (required): Display name.
- `email` (required): Email address.

## Errors

### 400 Bad Request

Invalid parameters or body.

### 401 Unauthorized

Missing or invalid API key.

### 404 Not Found

Resource does not exist.

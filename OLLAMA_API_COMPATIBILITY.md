# Ollama API Compatibility

This document outlines the Ollama API compatibility implemented in oai2ollama.

## Implemented Endpoints

### Core Ollama API Endpoints

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/api/tags` | GET | ✅ Implemented | List available models |
| `/api/show` | POST | ✅ Implemented | Show model information |
| `/api/version` | GET | ✅ Implemented | Get Ollama version |
| `/api/generate` | POST | ✅ Implemented | Generate text completion |
| `/api/chat` | POST | ✅ Implemented | Chat with model |
| `/api/embeddings` | POST | ✅ Implemented | Create embeddings |
| `/api/ps` | GET | ✅ Implemented | List running models |
| `/api/pull` | POST | ✅ Implemented | Pull model (mock) |
| `/api/delete` | DELETE | ✅ Implemented | Delete model (mock) |
| `/api/copy` | POST | ✅ Implemented | Copy model (mock) |
| `/api/create` | POST | ✅ Implemented | Create model (mock) |

### OpenAI-Compatible Endpoints

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/v1/models` | GET | ✅ Implemented | List models (OpenAI format) |
| `/v1/chat/completions` | POST | ✅ Implemented | Chat completions (OpenAI format) |

## API Format Conversion

The implementation handles conversion between Ollama and OpenAI API formats:

### Ollama → OpenAI Request Mapping
- `model` → `model`
- `prompt` → `prompt` (for generate)
- `messages` → `messages` (for chat)
- `options.num_predict` → `max_tokens`
- `options.temperature` → `temperature`
- `stream` → `stream`

### OpenAI → Ollama Response Mapping
- `choices[0].text` → `response` (for generate)
- `choices[0].message` → `message` (for chat)
- `data[0].embedding` → `embeddings` (for embeddings)

## Features

### Streaming Support
Both `/api/generate` and `/api/chat` endpoints support streaming responses in Ollama's NDJSON format.

### Thinking Support
When `THINKING_ENABLE=true`, the chat endpoints automatically add thinking support to OpenAI requests.

### Model Management
Model management endpoints (`/pull`, `/delete`, `/copy`, `/create`) provide mock implementations since they cannot be proxied through OpenAI APIs.

### Debug Logging
Enhanced debug logging with `DEBUG_API_CALLS=true` to monitor API requests and responses.

## Usage Examples

### Generate Text
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt": "Hello, world!",
    "stream": false
  }'
```

### Chat
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

### List Models
```bash
curl http://localhost:11434/api/tags
```

### Create Embeddings
```bash
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-ada-002",
    "prompt": "Hello, world!"
  }'
```

## Compatibility Notes

- The implementation maintains full compatibility with Ollama client libraries
- All core functionality is supported through proxying to OpenAI-compatible APIs
- Model management operations return appropriate responses but don't perform actual operations
- Embeddings fall back to mock responses if the OpenAI API doesn't support them
- The server runs on port 11434 by default (same as Ollama)
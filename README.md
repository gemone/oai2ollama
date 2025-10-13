# Oai2Ollama Go

A high-performance Go implementation of Oai2Ollama that provides Ollama API compatibility for OpenAI-compatible backends.

## 🚀 Features

- **🔄 API Compatibility**: Complete Ollama API compatibility layer
- **🌐 Multiple Backends**: Support for OpenAI and other OpenAI-compatible APIs
- **🏷️ Model Prefixing**: Automatic model name prefixing to avoid conflicts
- **📊 Streaming Support**: Full streaming support for real-time responses
- **📈 API Metrics**: Comprehensive token statistics and request monitoring
- **⚡ High Performance**: Built with GoFiber for maximum performance
- **🔧 Flexible Configuration**: YAML and environment variable configuration
- **📝 Rich Examples**: Comprehensive examples and documentation

## 📋 Quick Start

### Prerequisites

- Go 1.21 or later
- OpenAI API key (or other compatible API)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/gemone/oai2ollama.git
cd oai2ollama-go
```

2. Install dependencies:

```bash
make deps
```

3. Configure your API key:

```bash
make config
# Edit configs/config.yaml and replace 'sk-your-openai-api-key-here' with your actual API key
```

4. Build and run:

```bash
make run
```

Or run in development mode:

```bash
make dev
```

### Basic Usage

#### List Models

```bash
curl http://localhost:11434/api/tags
```

#### Chat Completion

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Streaming Chat

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

## � API Metrics & Token Statistics

- **Automatic Collection**: Metrics are collected automatically for all API requests
- **Token Tracking**: Prompt and completion tokens are tracked for each request
- **Historical Data**: Configurable retention period for metrics storage
- **Real-time Monitoring**: Live statistics and aggregated reports
- **Privacy Controls**: IP anonymization and configurable data collection

### Metrics Endpoints

```bash
# Get comprehensive metrics summary
curl http://localhost:11434/metrics/

# Get detailed request metrics with filters
curl "http://localhost:11434/metrics/requests?model=openai/gpt-4&limit=10"

# Get token usage statistics
curl http://localhost:11434/metrics/tokens

# Get model-specific metrics
curl http://localhost:11434/metrics/models/openai/gpt-4
```

### Testing Metrics

```bash
# Run the comprehensive test script
./scripts/test_metrics.sh
```

## �🛠️ Configuration

### Config File (`configs/config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 11434
  debug: false

metrics:
  enabled: true
  database_path: "metrics.db"
  retention_period: "720h"
  collect_request_size: true
  collect_response_size: true
  anonymize_ips: true

backends:
  - name: "openai"
    api_key: "sk-your-openai-api-key-here" # 替换为您的实际密钥
    base_url: "https://api.openai.com/v1"
    enabled: true
    timeout: 30
    priority: 1
    model_prefix:
      enabled: true
      prefix: "openai"
      separator: "/"

models:
  - name: "*"
    backend: "openai"
    capabilities: ["completion", "tools"]
    enabled: true
```

所有配置都在 `configs/config.yaml` 文件中管理，包括 API 密钥。

## 🎯 Model Prefixing

Oai2Ollama Go automatically adds prefixes to model names to prevent conflicts when using multiple backends:

- **OpenAI models**: `openai/gpt-4`, `openai/gpt-3.5-turbo`
- **Custom backends**: Use configurable prefixes like `claude/claude-3-sonnet`

## 📚 API Reference

### Ollama Compatible Endpoints

| Endpoint          | Method | Description              |
| ----------------- | ------ | ------------------------ |
| `/api/tags`       | GET    | List available models    |
| `/api/chat`       | POST   | Chat with a model        |
| `/api/generate`   | POST   | Generate text completion |
| `/api/embeddings` | POST   | Generate embeddings      |
| `/api/version`    | GET    | Get version info         |
| `/api/ps`         | GET    | Show running models      |
| `/api/pull`       | POST   | Pull a model (mock)      |
| `/api/delete`     | DELETE | Delete a model (mock)    |

### Response Format

All responses follow the standard Ollama API format, ensuring compatibility with existing tools and applications.

## 🧪 Testing

Run the test script to verify functionality:

```bash
make test
# or
./scripts/test.sh
```

## 🏗️ Development

### Build

```bash
make build
```

### Run in Development

```bash
make dev
```

### Run Tests

```bash
make test
```

### Build for Multiple Platforms

```bash
make build-all
```

### Code Quality

```bash
make fmt    # Format code
make lint   # Lint code
```

## 📁 Project Structure

```
oai2ollama-go/
├── cmd/oai2ollama/          # Application entry point
├── internal/
│   ├── config/              # Configuration management
│   ├── handlers/            # HTTP handlers
│   ├── models/              # Data models
│   └── services/            # Business logic
├── pkg/
│   ├── client/              # API clients
│   └── utils/               # Utilities
├── configs/                 # Configuration files
├── scripts/                 # Helper scripts
└── docs/                    # Documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Original Project](https://github.com/gemone/oai2ollama)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Ollama API Documentation](https://docs.ollama.com/api)
- [GoFiber Framework](https://docs.gofiber.io/)

## 🙏 Acknowledgments

- Original oai2ollama project for the concept and API design
- OpenAI for providing the powerful models
- Ollama team for the excellent API design
- GoFiber team for the amazing web framework


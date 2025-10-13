.PHONY: build run test clean deps

# Build the application
build:
	@echo "Building oai2ollama-go..."
	@mkdir -p bin
	@go build -o bin/oai2ollama-go ./cmd/oai2ollama

# Run the application
run: build
	@echo "Running oai2ollama-go..."
	@./bin/oai2ollama-go

# Run in development mode
dev:
	@echo "Running in development mode..."
	@go run ./cmd/oai2ollama

# Install dependencies
deps:
	@echo "Installing dependencies..."
	@go mod download
	@go mod tidy

# Run tests
test:
	@echo "Running tests..."
	@go test -v ./...

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf bin/
	@rm -f oai2ollama-go

# Build for multiple platforms
build-all:
	@echo "Building for multiple platforms..."
	@mkdir -p bin
	@GOOS=linux GOARCH=amd64 go build -o bin/oai2ollama-go-linux-amd64 ./cmd/oai2ollama
	@GOOS=darwin GOARCH=amd64 go build -o bin/oai2ollama-go-darwin-amd64 ./cmd/oai2ollama
	@GOOS=darwin GOARCH=arm64 go build -o bin/oai2ollama-go-darwin-arm64 ./cmd/oai2ollama
	@GOOS=windows GOARCH=amd64 go build -o bin/oai2ollama-go-windows-amd64.exe ./cmd/oai2ollama

# Docker build
docker-build:
	@echo "Building Docker image..."
	@docker build -t oai2ollama-go:latest .

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...

# Lint code
lint:
	@echo "Linting code..."
	@golangci-lint run

# Install development tools
install-tools:
	@echo "Installing development tools..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Quick start
start:
	@echo "Starting oai2ollama-go..."
	@echo "Please edit configs/config.yaml to set your API keys first"
	@echo "Example: edit configs/config.yaml and replace 'sk-your-openai-api-key-here' with your actual API key"
	@echo "Then run: make run"

# Edit configuration
config:
	@if command -v code > /dev/null 2>&1; then code configs/config.yaml; elif command -v nano > /dev/null 2>&1; then nano configs/config.yaml; elif command -v vim > /dev/null 2>&1; then vim configs/config.yaml; else echo "Please edit configs/config.yaml manually"; fi

# Help
help:
	@echo "Available commands:"
	@echo "  build      - Build the application"
	@echo "  run        - Build and run the application"
	@echo "  dev        - Run in development mode"
	@echo "  deps       - Install dependencies"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean build artifacts"
	@echo "  build-all  - Build for multiple platforms"
	@echo "  docker-build - Build Docker image"
	@echo "  fmt        - Format code"
	@echo "  lint       - Lint code"
	@echo "  install-tools - Install development tools"
	@echo "  start      - Quick start with environment setup"
	@echo "  help       - Show this help message"

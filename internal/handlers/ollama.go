package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/gemone/oai2ollama/internal/config"
	"github.com/gemone/oai2ollama/internal/middleware"
	"github.com/gemone/oai2ollama/internal/models"
	"github.com/gemone/oai2ollama/internal/services"
	"github.com/gemone/oai2ollama/pkg/client"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/log"
)

type OllamaHandler struct {
	converter      *services.ModelConverter
	backendClients map[string]models.BackendClient
	config         *config.Config
	modelCache     *services.ModelCache
	metricsService *services.MetricsService
}

func NewOllamaHandler(cfg *config.Config) *OllamaHandler {
	handler := &OllamaHandler{
		converter:      services.NewModelConverter(cfg),
		backendClients: make(map[string]models.BackendClient),
		config:         cfg,
	}

	// Initialize backend clients
	for _, backend := range cfg.Backends {
		if backend.Enabled {
			handler.backendClients[backend.Name] = client.NewOpenAIClient(&backend)
		}
	}

	// Initialize model cache
	handler.modelCache = services.NewModelCache(cfg, handler.backendClients, handler.converter)

	// Initialize metrics service
	metricsConfig := models.DefaultMetricsConfig()
	if cfg.Metrics.Enabled {
		metricsConfig.Enabled = cfg.Metrics.Enabled
		metricsConfig.DatabasePath = cfg.Metrics.DatabasePath
		metricsConfig.MaxConnections = cfg.Metrics.MaxConnections
		metricsConfig.CollectRequestSize = cfg.Metrics.CollectRequestSize
		metricsConfig.CollectResponseSize = cfg.Metrics.CollectResponseSize
		metricsConfig.CollectUserAgent = cfg.Metrics.CollectUserAgent
		metricsConfig.CollectClientIP = cfg.Metrics.CollectClientIP
		metricsConfig.AnonymizeIPs = cfg.Metrics.AnonymizeIPs

		// Parse duration strings
		if cfg.Metrics.RetentionPeriod != "" {
			if duration, err := time.ParseDuration(cfg.Metrics.RetentionPeriod); err == nil {
				metricsConfig.RetentionPeriod = duration
			}
		}
		if cfg.Metrics.AggregationInterval != "" {
			if duration, err := time.ParseDuration(cfg.Metrics.AggregationInterval); err == nil {
				metricsConfig.AggregationInterval = duration
			}
		}
	}

	metricsService, err := services.NewMetricsService(metricsConfig)
	if err != nil {
		log.Errorf("Failed to initialize metrics service: %v", err)
		// Continue without metrics if initialization fails
		handler.metricsService = nil
	} else {
		handler.metricsService = metricsService
	}

	return handler
}

// GetMetricsService returns the metrics service instance
func (h *OllamaHandler) GetMetricsService() *services.MetricsService {
	return h.metricsService
}

// GET /metrics - Get metrics summary
func (h *OllamaHandler) GetMetricsSummary(c *fiber.Ctx) error {
	if h.metricsService == nil {
		return c.Status(503).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Metrics service not available",
				Type:    "service_unavailable",
				Code:    "metrics_disabled",
			},
		})
	}

	// Parse query parameters
	filter := h.parseMetricsFilter(c)

	summary, err := h.metricsService.GetSummary(filter)
	if err != nil {
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: fmt.Sprintf("Failed to get metrics summary: %v", err),
				Type:    "internal_error",
				Code:    "metrics_error",
			},
		})
	}

	return c.JSON(summary)
}

// GET /metrics/requests - Get detailed metrics
func (h *OllamaHandler) GetMetrics(c *fiber.Ctx) error {
	if h.metricsService == nil {
		return c.Status(503).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Metrics service not available",
				Type:    "service_unavailable",
				Code:    "metrics_disabled",
			},
		})
	}

	// Parse query parameters
	filter := h.parseMetricsFilter(c)

	metrics, err := h.metricsService.GetMetrics(filter)
	if err != nil {
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: fmt.Sprintf("Failed to get metrics: %v", err),
				Type:    "internal_error",
				Code:    "metrics_error",
			},
		})
	}

	return c.JSON(map[string]interface{}{
		"metrics": metrics,
		"filter":  filter,
		"total":   len(metrics),
	})
}

// GET /metrics/tokens - Get token usage statistics
func (h *OllamaHandler) GetTokenUsage(c *fiber.Ctx) error {
	if h.metricsService == nil {
		return c.Status(503).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Metrics service not available",
				Type:    "service_unavailable",
				Code:    "metrics_disabled",
			},
		})
	}

	// Parse query parameters
	filter := h.parseMetricsFilter(c)

	tokenUsage, err := h.metricsService.GetTokenUsage(filter)
	if err != nil {
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: fmt.Sprintf("Failed to get token usage: %v", err),
				Type:    "internal_error",
				Code:    "metrics_error",
			},
		})
	}

	return c.JSON(map[string]interface{}{
		"token_usage": tokenUsage,
		"filter":      filter,
	})
}

// GET /metrics/models/:model - Get model-specific metrics
func (h *OllamaHandler) GetModelMetrics(c *fiber.Ctx) error {
	if h.metricsService == nil {
		return c.Status(503).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Metrics service not available",
				Type:    "service_unavailable",
				Code:    "metrics_disabled",
			},
		})
	}

	modelName := c.Params("model")
	if modelName == "" {
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Model name is required",
				Type:    "invalid_request_error",
				Code:    "missing_model",
				Param:   "model",
			},
		})
	}

	// Parse query parameters
	filter := h.parseMetricsFilter(c)

	summary, err := h.metricsService.GetMetricsByModel(modelName, filter)
	if err != nil {
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: fmt.Sprintf("Failed to get model metrics: %v", err),
				Type:    "internal_error",
				Code:    "metrics_error",
			},
		})
	}

	return c.JSON(map[string]interface{}{
		"model":   modelName,
		"metrics": summary,
	})
}

// GET /metrics/backends/:backend - Get backend-specific metrics
func (h *OllamaHandler) GetBackendMetrics(c *fiber.Ctx) error {
	if h.metricsService == nil {
		return c.Status(503).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Metrics service not available",
				Type:    "service_unavailable",
				Code:    "metrics_disabled",
			},
		})
	}

	backendName := c.Params("backend")
	if backendName == "" {
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Backend name is required",
				Type:    "invalid_request_error",
				Code:    "missing_backend",
				Param:   "backend",
			},
		})
	}

	// Parse query parameters
	filter := h.parseMetricsFilter(c)

	summary, err := h.metricsService.GetMetricsByBackend(backendName, filter)
	if err != nil {
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: fmt.Sprintf("Failed to get backend metrics: %v", err),
				Type:    "internal_error",
				Code:    "metrics_error",
			},
		})
	}

	return c.JSON(map[string]interface{}{
		"backend": backendName,
		"metrics": summary,
	})
}

// parseMetricsFilter parses query parameters into a MetricsFilter
func (h *OllamaHandler) parseMetricsFilter(c *fiber.Ctx) models.MetricsFilter {
	filter := models.MetricsFilter{}

	// Parse time range
	if startTimeStr := c.Query("start_time"); startTimeStr != "" {
		if startTime, err := time.Parse(time.RFC3339, startTimeStr); err == nil {
			filter.StartTime = &startTime
		}
	}

	if endTimeStr := c.Query("end_time"); endTimeStr != "" {
		if endTime, err := time.Parse(time.RFC3339, endTimeStr); err == nil {
			filter.EndTime = &endTime
		}
	}

	// Parse filters
	filter.Model = c.Query("model")
	filter.Backend = c.Query("backend")
	filter.Method = c.Query("method")

	if statusStr := c.Query("status"); statusStr != "" {
		if status, err := strconv.Atoi(statusStr); err == nil {
			filter.Status = &status
		}
	}

	if minTokensStr := c.Query("min_tokens"); minTokensStr != "" {
		if minTokens, err := strconv.Atoi(minTokensStr); err == nil {
			filter.MinTokens = &minTokens
		}
	}

	if maxTokensStr := c.Query("max_tokens"); maxTokensStr != "" {
		if maxTokens, err := strconv.Atoi(maxTokensStr); err == nil {
			filter.MaxTokens = &maxTokens
		}
	}

	if streamingStr := c.Query("streaming"); streamingStr != "" {
		if streaming, err := strconv.ParseBool(streamingStr); err == nil {
			filter.Streaming = &streaming
		}
	}

	// Parse pagination
	if limitStr := c.Query("limit"); limitStr != "" {
		if limit, err := strconv.Atoi(limitStr); err == nil && limit > 0 {
			filter.Limit = limit
		}
	}

	if offsetStr := c.Query("offset"); offsetStr != "" {
		if offset, err := strconv.Atoi(offsetStr); err == nil && offset >= 0 {
			filter.Offset = offset
		}
	}

	// Parse sorting
	filter.OrderBy = c.Query("order_by", "timestamp")
	filter.OrderDir = c.Query("order_dir", "desc")

	return filter
}

// GET /api/tags - List models
func (h *OllamaHandler) ListModels(c *fiber.Ctx) error {
	// Get all models from cache
	allModels := h.modelCache.GetAllModels()

	response := models.OllamaModelsResponse{
		Models: allModels,
	}

	return c.JSON(response)
}

// POST /api/chat - Chat completion
func (h *OllamaHandler) Chat(c *fiber.Ctx) error {
	// Parse Ollama request
	var request models.OllamaChatRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Invalid request format",
				Type:    "invalid_request_error",
				Code:    "invalid_format",
			},
		})
	}

	// Convert Ollama request to OpenAI format
	openAIRequest := h.convertOllamaToOpenAIRequest(&request, request.Model)

	// Process using universal function (return Ollama format)
	return h.processChatCompletion(c, openAIRequest, false)
}

// convertOllamaToOpenAIRequest converts Ollama chat request to OpenAI format
func (h *OllamaHandler) convertOllamaToOpenAIRequest(request *models.OllamaChatRequest, originalModel string) *models.OpenAIChatCompletionRequest {
	openAIRequest := &models.OpenAIChatCompletionRequest{
		Model:    originalModel, // Use original name for backend API call
		Messages: make([]models.OpenAIMessage, len(request.Messages)),
		Stream:   request.Stream,
	}

	// Add parameters only if they are not nil/empty
	if request.Options != nil {
		// Only add temperature if it's not nil
		if request.Options.Temperature != nil {
			openAIRequest.Temperature = request.Options.Temperature
		}

		// Only add top_p if it's not nil
		if request.Options.TopP != nil {
			openAIRequest.TopP = request.Options.TopP
		}

		// Only add max_tokens if it's not nil
		if request.Options.NumPredict != nil {
			openAIRequest.MaxTokens = request.Options.NumPredict
		}

		// Handle stop sequences
		if len(request.Options.Stop) > 0 {
			openAIRequest.Stop = request.Options.Stop
		}

		// Handle presence penalty
		if request.Options.PresencePenalty != nil {
			openAIRequest.PresencePenalty = request.Options.PresencePenalty
		}

		// Handle frequency penalty
		if request.Options.FrequencyPenalty != nil {
			openAIRequest.FrequencyPenalty = request.Options.FrequencyPenalty
		}
	}

	// Convert messages
	for i, msg := range request.Messages {
		openAIRequest.Messages[i] = models.OllamaToOpenAIMessage(msg)
	}

	// Convert Ollama think parameter to OpenAI thinking format
	if request.Think {
		openAIRequest.Thinking = &models.OpenAIThinking{
			Type: "enabled",
		}
	}

	return openAIRequest
}

// POST /api/generate - Text generation
func (h *OllamaHandler) Generate(c *fiber.Ctx) error {
	var request models.OllamaGenerateRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(400).JSON(models.OllamaErrorResponse{
			Error: "Invalid request format",
		})
	}

	log.Debugf("Request body: %s", string(c.Body()))

	// Parse model name to get original name for backend call
	// log.Debugf("Request body: %s", string(c.Body())) // Removed to avoid logging sensitive information
	if h.converter == nil {
		return c.Status(500).JSON(models.OllamaErrorResponse{
			Error: "Internal server error: model converter not initialized",
		})
	}
	modelParse, err := h.converter.ParseModelName(request.Model)
	if err != nil {
		return c.Status(400).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Invalid model name: %s", request.Model),
		})
	}

	log.Debugf("Generate model parse result: Backend=%s, OriginalName=%s, DisplayName=%s",
		modelParse.Backend, modelParse.OriginalName, modelParse.DisplayName)

	// Validate and set prompt using helper
	prompt, err := h.getValidatedPrompt(request.Prompt, modelParse.Backend)
	if err != nil {
		log.Debugf("Prompt validation failed: %v", err)
		return c.Status(400).JSON(models.OllamaErrorResponse{
			Error: err.Error(),
		})
	}
	request.Prompt = prompt

	// Use the chat handler logic with converted model name
	chatHandlerRequest := &models.OpenAIChatCompletionRequest{
		Model:    modelParse.OriginalName, // Use original name for backend
		Messages: []models.OpenAIMessage{{Role: "user", Content: request.Prompt}},
		Stream:   request.Stream,
	}

	// Safely extract options if they exist
	if request.Options != nil {
		if request.Options.Temperature != nil {
			chatHandlerRequest.Temperature = request.Options.Temperature
		}
		if request.Options.TopP != nil {
			chatHandlerRequest.TopP = request.Options.TopP
		}
		if request.Options.NumPredict != nil {
			chatHandlerRequest.MaxTokens = request.Options.NumPredict
		}
	}

	// Convert Ollama think parameter to OpenAI thinking format
	if request.Think {
		chatHandlerRequest.Thinking = &models.OpenAIThinking{
			Type: "enabled",
		}
	}

	// Get backend for this model using cache
	backendName, err := h.modelCache.GetBackendForModel(request.Model)
	if err != nil {
		return c.Status(404).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Model not found: %s", request.Model),
		})
	}

	backendClient, exists := h.backendClients[backendName]
	if !exists {
		return c.Status(503).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Backend not available: %s", backendName),
		})
	}

	// Handle streaming
	if request.Stream {
		return h.handleStreamingGenerate(c, backendClient, chatHandlerRequest)
	}

	// Handle non-streaming
	response, err := backendClient.ChatCompletion(chatHandlerRequest)
	if err != nil {
		return c.Status(500).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Generation failed: %v", err),
		})
	}

	// Convert to generate format
	generateResponse := models.OllamaGenerateResponse{
		Model:    request.Model,
		Response: response.Choices[0].Message.Content.(string),
		Done:     true,
		Context:  request.Context,
	}

	if response.Usage != nil {
		generateResponse.PromptEvalCount = response.Usage.PromptTokens
		generateResponse.EvalCount = response.Usage.CompletionTokens

		// Record token usage in response headers for middleware
		middleware.SetTokenUsage(c, response.Usage.PromptTokens,
			response.Usage.CompletionTokens, response.Usage.TotalTokens)
	}

	return c.JSON(generateResponse)
}

func (h *OllamaHandler) handleStreamingGenerate(c *fiber.Ctx, backendClient models.BackendClient, request *models.OpenAIChatCompletionRequest) error {
	// Set SSE headers
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")

	// Start streaming (thinking detection is handled inside the client)
	ch, err := backendClient.StreamChatCompletion(c.Context(), request)

	if err != nil {
		return c.Status(500).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Failed to start stream: %v", err),
		})
	}
	// Stream responses
	for response := range ch {
		if len(response.Choices) > 0 {
			if contentStr, ok := response.Choices[0].Delta.Content.(string); ok && contentStr != "" {
				// Convert to generate format
				generateResponse := models.OllamaGenerateResponse{
					Model:     request.Model,
					CreatedAt: time.Now(),
					Response:  contentStr,
					Done:      false,
				}

				// Send as SSE
				data := fmt.Sprintf("data: %s\n\n", mustMarshalJSON(generateResponse))
				if _, err := c.WriteString(data); err != nil {
					return fmt.Errorf("failed to write SSE data: %w", err)
				}
			}
		}

		if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
			// Send final chunk
			finalResponse := &models.OllamaGenerateResponse{
				Model:     request.Model,
				CreatedAt: time.Now(),
				Done:      true,
			}
			finalData := fmt.Sprintf("data: %s\n\n", mustMarshalJSON(finalResponse))
			if _, err := c.WriteString(finalData); err != nil {
				return fmt.Errorf("failed to write final SSE data: %w", err)
			}
			break
		}
	}

	// Send done signal
	if _, err := c.WriteString("data: [DONE]\n\n"); err != nil {
		return fmt.Errorf("failed to write SSE done signal: %w", err)
	}

	return nil
}

// POST /api/embeddings - Generate embeddings
func (h *OllamaHandler) Embeddings(c *fiber.Ctx) error {
	var request models.OllamaEmbeddingsRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Invalid request format",
				Type:    "invalid_request_error",
				Code:    "invalid_format",
			},
		})
	}

	// Validate request
	if request.Model == "" {
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Model is required",
				Type:    "invalid_request_error",
				Code:    "missing_model",
				Param:   "model",
			},
		})
	}

	if request.Prompt == "" {
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Prompt is required",
				Type:    "invalid_request_error",
				Code:    "missing_prompt",
				Param:   "prompt",
			},
		})
	}

	// Get backend for this model using cache
	backendName, err := h.modelCache.GetBackendForModel(request.Model)
	if err != nil {
		return c.Status(404).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Model not found: %s", request.Model),
		})
	}

	backendClient, exists := h.backendClients[backendName]
	if !exists {
		return c.Status(503).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Backend not available: %s", backendName),
		})
	}

	// Generate embeddings
	response, err := backendClient.GenerateEmbeddings(&request)
	if err != nil {
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: fmt.Sprintf("Embeddings generation failed: %v", err),
				Type:    "api_error",
				Code:    "embeddings_failed",
			},
		})
	}

	return c.JSON(response)
}

// GET /api/version - Get version
func (h *OllamaHandler) Version(c *fiber.Ctx) error {
	response := map[string]interface{}{
		"version": "0.12.5",
	}

	return c.JSON(response)
}

// GET /api/ps - Show running models (mock)
func (h *OllamaHandler) ShowRunningModels(c *fiber.Ctx) error {
	// Mock response - in a real implementation, this would track running models
	response := map[string]interface{}{
		"models": []interface{}{},
	}

	return c.JSON(response)
}

// Mock endpoints for model management
func (h *OllamaHandler) PullModel(c *fiber.Ctx) error {
	var request struct {
		Name   string `json:"name"`
		Stream bool   `json:"stream"`
	}

	if err := c.BodyParser(&request); err != nil {
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Invalid request format",
				Type:    "invalid_request_error",
				Code:    "invalid_format",
			},
		})
	}

	// Mock response - this is a no-op since we're proxying to OpenAI
	response := map[string]interface{}{
		"status": "pulling " + request.Name,
	}

	return c.JSON(response)
}

func (h *OllamaHandler) DeleteModel(c *fiber.Ctx) error {
	var request struct {
		Name string `json:"name"`
	}

	if err := c.BodyParser(&request); err != nil {
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Invalid request format",
				Type:    "invalid_request_error",
				Code:    "invalid_format",
			},
		})
	}

	// Mock response - this is a no-op since we're proxying to OpenAI
	response := map[string]interface{}{
		"status": "deleted " + request.Name,
	}

	return c.JSON(response)
}

// POST /api/show - Show model information
func (h *OllamaHandler) ShowModel(c *fiber.Ctx) error {
	var request models.OllamaShowRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(400).JSON(models.OllamaErrorResponse{
			Error: "Invalid request format",
		})
	}
	if request.Name != "" && request.Model == "" {
		request.Model = request.Name
	}

	log.Debugf("Show model request: %s (verbose: %t)", request.Model, request.Verbose)

	// Validate request
	if request.Model == "" {
		return c.Status(400).JSON(models.OllamaErrorResponse{
			Error: "Model name is required",
		})
	}

	// Get model from cache
	model, exists := h.modelCache.GetModel(request.Model)
	if !exists {
		return c.Status(404).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Model not found: %s", request.Model),
		})
	}

	log.Debugf("Found model: %s", model.Name)

	// Validate model details to prevent nil pointer dereference
	if model.Details == nil {
		log.Debugf("Model details is nil for model: %s", request.Model)
		model.Details = make(map[string]interface{})
	}

	// Parse model name to determine family and generate appropriate modelfile
	modelParse, err := h.converter.ParseModelName(request.Model)
	if err != nil {
		log.Debugf("Failed to parse model name: %v", err)
		modelParse = &models.ModelParseResult{
			Backend:      "unknown",
			OriginalName: request.Model,
			DisplayName:  request.Model,
		}
	}

	// Generate modelfile based on model family
	var modelfile string
	var template string
	var parameters string

	family := "llama"
	if familyVal, ok := model.Details["family"].(string); ok {
		family = strings.ToLower(familyVal)
	}

	switch family {
	case "llama", "llama3":
		modelfile = fmt.Sprintf(`# Modelfile generated by "ollama show"
# To build a new Modelfile based on this one, replace the FROM line with:
# FROM %s:latest

FROM %s
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER num_ctx 4096
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
`, modelParse.OriginalName, modelParse.OriginalName)

		template = "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>"

		parameters = `num_keep                       24
stop                           "<|start_header_id|>"
stop                           "<|end_header_id|>"
stop                           "<|eot_id|>"
num_ctx                        4096`

	case "gpt":
		modelfile = fmt.Sprintf(`# Modelfile generated by "ollama show"
FROM %s
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
PARAMETER num_ctx 4096
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
`, modelParse.OriginalName)

		template = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>"

		parameters = `num_ctx                        4096
stop                           "<|im_start|>"
stop                           "<|im_end|>"
stop                           "<|im_sep|>"`

	default:
		modelfile = fmt.Sprintf(`# Modelfile generated by "ollama show"
FROM %s
TEMPLATE """{{ if .System }}System: {{ .System }}

{{ end }}User: {{ .Prompt }}
Assistant: """
PARAMETER num_ctx 4096
`, modelParse.OriginalName)

		template = "{{ if .System }}System: {{ .System }}\n\n{{ end }}User: {{ .Prompt }}\nAssistant: "

		parameters = `num_ctx                        4096
stop                           "User:"
stop                           "Assistant:"`
	}

	// Determine capabilities based on model and backend
	var capabilities []string
	if caps, ok := model.Details["capabilities"].([]string); ok {
		capabilities = caps
	} else {
		// Default capabilities
		capabilities = []string{"completion"}
	}

	// Create response with model information
	response := &models.OllamaShowResponse{
		Modelfile:    modelfile,
		Parameters:   parameters,
		Template:     template,
		Details:      model.Details,
		ModifiedAt:   model.ModifiedAt,
		Capabilities: capabilities,
	}

	// Always include model_info with default CausalLM architecture
	backendName := modelParse.Backend
	if backend, ok := model.Details["backend"].(string); ok && backend != "" {
		backendName = backend
	}

	// Get architecture based on family
	architecture := family
	if architecture == "" {
		architecture = "CausalLM"
	}

	response.ModelInfo = map[string]interface{}{
		"general.architecture": architecture,
	}

	// Add additional info if verbose is requested
	if request.Verbose {
		generalInfo := map[string]interface{}{
			"architecture": architecture,
			"file_type":    2, // GGUF
		}

		response.ModelInfo = map[string]interface{}{
			"general": generalInfo,
			"backend": map[string]interface{}{
				"name": backendName,
			},
		}
	}

	log.Debugf("Returning show information for model: %s", request.Model)

	return c.JSON(response)
}

// Helper function
func mustMarshalJSON(v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return data
}

// Universal chat completion processing function
func (h *OllamaHandler) processChatCompletion(c *fiber.Ctx, request *models.OpenAIChatCompletionRequest, returnOpenAIFormat bool) error {
	log.Debug("ProcessChatCompletion called")

	log.Debugf("Parsed request: Model=%s, MessagesCount=%d, Stream=%t",
		request.Model, len(request.Messages), request.Stream)

	// Validate request
	if request.Model == "" {
		log.Debug("Model validation failed - empty model")
		if returnOpenAIFormat {
			return c.Status(400).JSON(models.ErrorResponse{
				Error: models.APIError{
					Message: "Model is required",
					Type:    "invalid_request_error",
					Code:    "missing_model",
					Param:   "model",
				},
			})
		} else {
			return c.Status(400).JSON(models.OllamaErrorResponse{
				Error: "Model name is required",
			})
		}
	}

	if len(request.Messages) == 0 {
		log.Debug("Messages validation failed - no messages")
		if returnOpenAIFormat {
			return c.Status(400).JSON(models.ErrorResponse{
				Error: models.APIError{
					Message: "Messages are required",
					Type:    "invalid_request_error",
					Code:    "missing_messages",
					Param:   "messages",
				},
			})
		} else {
			return c.Status(400).JSON(models.OllamaErrorResponse{
				Error: "Messages are required",
			})
		}
	}

	// Get backend for this model using cache
	log.Debugf("Looking up backend for model: %s", request.Model)
	backendName, err := h.modelCache.GetBackendForModel(request.Model)
	if err != nil {
		log.Debugf("Backend lookup failed: %v", err)
		if returnOpenAIFormat {
			return c.Status(404).JSON(models.ErrorResponse{
				Error: models.APIError{
					Message: fmt.Sprintf("Model not found: %s", request.Model),
					Type:    "invalid_request_error",
					Code:    "model_not_found",
					Param:   "model",
				},
			})
		} else {
			return c.Status(404).JSON(models.OllamaErrorResponse{
				Error: fmt.Sprintf("Model not found: %s", request.Model),
			})
		}
	}

	log.Debugf("Found backend: %s", backendName)

	backendClient, exists := h.backendClients[backendName]
	if !exists {
		log.Debugf("Backend client not found for: %s", backendName)
		if returnOpenAIFormat {
			return c.Status(503).JSON(models.ErrorResponse{
				Error: models.APIError{
					Message: fmt.Sprintf("Backend not available: %s", backendName),
					Type:    "api_error",
					Code:    "backend_unavailable",
				},
			})
		} else {
			return c.Status(503).JSON(models.OllamaErrorResponse{
				Error: fmt.Sprintf("Backend not available: %s", backendName),
			})
		}
	}

	log.Debug("Backend client found, proceeding with request")

	// Parse model name to get original name for backend call
	log.Debugf("Parsing model name: %s", request.Model)
	if h.converter == nil {
		log.Error("Model converter is nil")
		if returnOpenAIFormat {
			return c.Status(500).JSON(models.ErrorResponse{
				Error: models.APIError{
					Message: "Internal server error: model converter not initialized",
					Type:    "internal_error",
					Code:    "internal_error",
				},
			})
		} else {
			return c.Status(500).JSON(models.OllamaErrorResponse{
				Error: "Internal server error: model converter not initialized",
			})
		}
	}

	modelParse, err := h.converter.ParseModelName(request.Model)
	if err != nil {
		log.Debugf("Model parsing failed: %v", err)
		if returnOpenAIFormat {
			return c.Status(400).JSON(models.ErrorResponse{
				Error: models.APIError{
					Message: fmt.Sprintf("Invalid model name: %s", request.Model),
					Type:    "invalid_request_error",
					Code:    "invalid_model",
					Param:   "model",
				},
			})
		} else {
			return c.Status(400).JSON(models.OllamaErrorResponse{
				Error: fmt.Sprintf("Invalid model name: %s", request.Model),
			})
		}
	}

	log.Debugf("Model parse result: Backend=%s, OriginalName=%s, DisplayName=%s",
		modelParse.Backend, modelParse.OriginalName, modelParse.DisplayName)

	// Store original prefixed model name for response
	originalPrefixedModel := request.Model
	// Update request model to use original name for backend API call
	request.Model = modelParse.OriginalName

	// Handle streaming
	if request.Stream {
		log.Debug("Handling streaming request")
		return h.handleStreamingChatCompletions(c, backendClient, request, returnOpenAIFormat, originalPrefixedModel)
	}

	// Handle non-streaming
	log.Debug("Handling non-streaming request")
	response, err := backendClient.ChatCompletion(request)
	if err != nil {
		log.Debugf("ChatCompletion failed: %v", err)
		if returnOpenAIFormat {
			return c.Status(500).JSON(models.ErrorResponse{
				Error: models.APIError{
					Message: fmt.Sprintf("Chat completion failed: %v", err),
					Type:    "api_error",
					Code:    "chat_completion_failed",
				},
			})
		} else {
			return c.Status(500).JSON(models.OllamaErrorResponse{
				Error: fmt.Sprintf("Generation failed: %v", err),
			})
		}
	}

	log.Debug("ChatCompletion succeeded")

	// Record token usage in response headers for middleware
	if response.Usage != nil {
		middleware.SetTokenUsage(c, response.Usage.PromptTokens,
			response.Usage.CompletionTokens, response.Usage.TotalTokens)
	}

	if returnOpenAIFormat {
		// For OpenAI format, restore the original model name in response
		response.Model = originalPrefixedModel
		return c.JSON(response)
	} else {
		// Convert OpenAI response to Ollama format
		ollamaResponse := &models.OllamaChatResponse{
			Model:     originalPrefixedModel, // Use prefixed model for Ollama format
			CreatedAt: time.Now(),
			Message:   models.OpenAIToOllamaMessage(response.Choices[0].Message),
			Done:      true,
		}

		// Add usage information if available
		if response.Usage != nil {
			ollamaResponse.PromptEvalCount = response.Usage.PromptTokens
			ollamaResponse.EvalCount = response.Usage.CompletionTokens
		}

		return c.JSON(ollamaResponse)
	}
}

// POST /v1/chat/completions - OpenAI compatible chat completions
func (h *OllamaHandler) ChatCompletions(c *fiber.Ctx) error {
	log.Debug("ChatCompletions handler called")

	var request models.OpenAIChatCompletionRequest
	if err := c.BodyParser(&request); err != nil {
		log.Debugf("BodyParser failed: %v", err)
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Invalid request format",
				Type:    "invalid_request_error",
				Code:    "invalid_format",
			},
		})
	}

	log.Debugf("Request Body: %s", string(c.Body()))

	// Use universal processing function with OpenAI format
	return h.processChatCompletion(c, &request, true)
}

func (h *OllamaHandler) handleStreamingChatCompletions(c *fiber.Ctx, backendClient models.BackendClient, request *models.OpenAIChatCompletionRequest, returnOpenAIFormat bool, prefixedModel ...string) error {
	// Generate unique request ID for tracking
	requestID := fmt.Sprintf("stream-%d", time.Now().UnixNano())
	log.Debugf("[%s] Starting streaming request for model: %s", requestID, request.Model)

	// Set proper SSE headers
	c.Set("Content-Type", "text/event-stream; charset=utf-8")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")
	c.Set("Access-Control-Allow-Headers", "Cache-Control")
	c.Set("X-Accel-Buffering", "no") // Disable nginx buffering

	// Get the underlying fasthttp response
	resp := c.Response()
	// Ensure no content encoding
	resp.Header.Del("Content-Encoding")

	// Determine which model name to use in responses
	modelInResponse := request.Model
	if len(prefixedModel) > 0 && !returnOpenAIFormat {
		modelInResponse = prefixedModel[0] // Use prefixed model for Ollama format
	}

	// Create a cancellable context for the backend request
	ctx, cancel := context.WithCancel(c.Context())

	// Check for client disconnection in a separate goroutine
	go func() {
		<-c.Context().Done() // Wait for client to disconnect
		log.Debugf("[%s] Client disconnected, cancelling backend request", requestID)
		cancel()
	}()

	// Start streaming with cancellable context
	log.Debugf("[%s] Initiating backend streaming request", requestID)
	ch, err := backendClient.StreamChatCompletion(ctx, request)
	if err != nil {
		log.Debugf("[%s] Failed to start stream: %v", requestID, err)
		if returnOpenAIFormat {
			return c.Status(500).JSON(models.ErrorResponse{
				Error: models.APIError{
					Message: fmt.Sprintf("Failed to start stream: %v", err),
					Type:    "api_error",
					Code:    "stream_failed",
				},
			})
		} else {
			return c.Status(500).JSON(models.OllamaErrorResponse{
				Error: fmt.Sprintf("Failed to start stream: %v", err),
			})
		}
	}

	log.Debugf("[%s] Backend streaming started successfully", requestID)

	// Enable streaming mode for fasthttp
	resp.SetBodyStreamWriter(func(w *bufio.Writer) {
		// Stream responses
		chunkCount := 0
		streamEndedNormally := false

		for {
			select {
			case <-ctx.Done():
				// Context was cancelled (client disconnected)
				log.Debugf("[%s] Streaming cancelled by client or context (processed %d chunks)", requestID, chunkCount)
				return

			case response, ok := <-ch:
				if !ok {
					// Channel closed, stream ended normally
					log.Debugf("[%s] Backend stream channel closed (processed %d chunks)", requestID, chunkCount)
					streamEndedNormally = true
					goto streamEnded
				}

				chunkCount++
				if config.GlobalConfig != nil && config.GlobalConfig.Server.Debug {
					log.Debugf("[%s] Received chunk %d from backend", requestID, chunkCount)
				}

				var data string
				if returnOpenAIFormat {
					// For OpenAI format, create a copy and restore the original model name
					responseCopy := response
					if len(prefixedModel) > 0 {
						responseCopy.Model = prefixedModel[0]
					}
					// Send as SSE in OpenAI format
					data = fmt.Sprintf("data: %s\n\n", mustMarshalJSON(responseCopy))
				} else {
					// Convert to Ollama format for /api/chat compatibility
					if len(response.Choices) > 0 && response.Choices[0].Delta != nil {
						ollamaResponse := &models.OllamaChatResponse{
							Model:     modelInResponse,
							CreatedAt: time.Now(),
							Message:   models.OpenAIToOllamaMessage(*response.Choices[0].Delta),
							Done:      false,
						}

						data = fmt.Sprintf("data: %s\n\n", mustMarshalJSON(ollamaResponse))
					} else {
						// Skip if no delta content
						continue
					}
				}

				// Write data directly to stream
				if _, err := w.WriteString(data); err != nil {
					log.Debugf("[%s] Failed to write streaming response for chunk %d: %v", requestID, chunkCount, err)
					return
				}

				// Flush to ensure immediate delivery to client
				if err := w.Flush(); err != nil {
					log.Debugf("[%s] Failed to flush streaming response for chunk %d: %v", requestID, chunkCount, err)
					return
				}

				// If this is the final chunk, send done signal
				if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
					log.Debugf("[%s] Received final chunk with finish reason: %s (total chunks: %d)",
						requestID, response.Choices[0].FinishReason, chunkCount)

					// Record token usage if available in the final chunk
					if response.Usage != nil {
						middleware.SetTokenUsage(c, response.Usage.PromptTokens,
							response.Usage.CompletionTokens, response.Usage.TotalTokens)
					}

					var doneData string
					if returnOpenAIFormat {
						// OpenAI format uses [DONE] marker
						doneData = "data: [DONE]\n\n"
					} else {
						// Send final chunk for Ollama format
						finalResponse := &models.OllamaChatResponse{
							Model:     modelInResponse,
							CreatedAt: time.Now(),
							Done:      true,
						}

						doneData = fmt.Sprintf("data: %s\n\n", mustMarshalJSON(finalResponse))
					}

					if _, err := w.WriteString(doneData); err != nil {
						log.Debugf("[%s] Failed to write final streaming response: %v", requestID, err)
					} else {
						// Final flush
						w.Flush()
					}
					log.Debugf("[%s] Streaming completed successfully (total chunks: %d)", requestID, chunkCount)
					return
				}
			}
		}

	streamEnded:
		// Handle case where stream ended normally but no chunks were processed
		if streamEndedNormally && chunkCount == 0 {
			log.Debugf("[%s] Stream ended with 0 chunks - falling back to non-streaming request", requestID)

			// Make a non-streaming request to get the complete response
			nonStreamingRequest := *request
			nonStreamingRequest.Stream = false

			response, err := backendClient.ChatCompletion(&nonStreamingRequest)
			if err != nil {
				log.Debugf("[%s] Fallback non-streaming request failed: %v", requestID, err)
				// Send error response
				if returnOpenAIFormat {
					errorData := fmt.Sprintf("data: %s\n\n", mustMarshalJSON(models.ErrorResponse{
						Error: models.APIError{
							Message: fmt.Sprintf("Streaming not supported and fallback failed: %v", err),
							Type:    "api_error",
							Code:    "streaming_fallback_failed",
						},
					}))
					if _, writeErr := w.WriteString(errorData); writeErr != nil {
						log.Debugf("Failed to write error data: %v", writeErr)
					}
				} else {
					errorData := fmt.Sprintf("data: %s\n\n", mustMarshalJSON(models.OllamaErrorResponse{
						Error: fmt.Sprintf("Streaming not supported and fallback failed: %v", err),
					}))
					if _, writeErr := w.WriteString(errorData); writeErr != nil {
						log.Debugf("Failed to write error data: %v", writeErr)
					}
				}
				w.Flush()
				return
			}

			// Stream the complete response as a single chunk
			if returnOpenAIFormat {
				// For OpenAI format, restore the original model name
				responseCopy := *response
				if len(prefixedModel) > 0 {
					responseCopy.Model = prefixedModel[0]
				}
				data := fmt.Sprintf("data: %s\n\n", mustMarshalJSON(responseCopy))
				if _, err := w.WriteString(data); err != nil {
					log.Debugf("[%s] Failed to write fallback response: %v", requestID, err)
				} else {
					w.Flush()
					log.Debugf("[%s] Sent fallback response as single chunk", requestID)
				}

				// Send done signal
				doneData := "data: [DONE]\n\n"
				if _, err := w.WriteString(doneData); err != nil {
					log.Debugf("[%s] Failed to write fallback done signal: %v", requestID, err)
				} else {
					w.Flush()
				}
			} else {
				// Convert to Ollama format
				if len(response.Choices) > 0 {
					ollamaResponse := &models.OllamaChatResponse{
						Model:     modelInResponse,
						CreatedAt: time.Now(),
						Message:   models.OpenAIToOllamaMessage(response.Choices[0].Message),
						Done:      true,
					}
					data := fmt.Sprintf("data: %s\n\n", mustMarshalJSON(ollamaResponse))
					if _, err := w.WriteString(data); err != nil {
						log.Debugf("[%s] Failed to write fallback Ollama response: %v", requestID, err)
					} else {
						w.Flush()
						log.Debugf("[%s] Sent fallback Ollama response as single chunk", requestID)
					}
				}
			}
		}
	})

	return nil
}

// Helper to validate prompt and apply default if needed
func (h *OllamaHandler) getValidatedPrompt(prompt string, backend string) (string, error) {
	if strings.TrimSpace(prompt) != "" {
		return prompt, nil
	}
	log.Warnf("Empty prompt received for backend %s, checking for default prompt", backend)
	defaultPrompt, hasDefaultPrompt := config.GetDefaultPromptForBackend(backend)
	if hasDefaultPrompt {
		log.Infof("Using default prompt for backend %s: %s", backend, defaultPrompt)
		return defaultPrompt, nil
	}
	log.Debugf("No default prompt configured for backend %s", backend)
	return "", fmt.Errorf("prompt cannot be empty and no default prompt configured for backend")
}

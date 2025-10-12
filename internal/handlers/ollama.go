package handlers

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gemone/oai2ollama/internal/config"
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

	return handler
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
	// Add debug logging at function start
	log.Debug("Chat handler called")

	var request models.OllamaChatRequest
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

	log.Debugf("Parsed request: Model=%s, MessagesCount=%d, Stream=%t, Think=%t",
		request.Model, len(request.Messages), request.Stream, request.Think)

	// Validate request
	if request.Model == "" {
		log.Debug("Model validation failed - empty model")
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Model is required",
				Type:    "invalid_request_error",
				Code:    "missing_model",
				Param:   "model",
			},
		})
	}

	if len(request.Messages) == 0 {
		log.Debug("Messages validation failed - no messages")
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Messages are required",
				Type:    "invalid_request_error",
				Code:    "missing_messages",
				Param:   "messages",
			},
		})
	}

	// Get backend for this model using cache
	log.Debugf("Looking up backend for model: %s", request.Model)
	backendName, err := h.modelCache.GetBackendForModel(request.Model)
	if err != nil {
		log.Debugf("Backend lookup failed: %v", err)
		return c.Status(404).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Model not found: %s", request.Model),
		})
	}

	log.Debugf("Found backend: %s", backendName)

	backendClient, exists := h.backendClients[backendName]
	if !exists {
		log.Debugf("Backend client not found for: %s", backendName)
		return c.Status(503).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Backend not available: %s", backendName),
		})
	}

	log.Debug("Backend client found, proceeding with request conversion")

	// Parse model name to get original name for backend call
	log.Debugf("Parsing model name: %s", request.Model)
	if h.converter == nil {
		log.Error("Model converter is nil")
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Internal server error: model converter not initialized",
				Type:    "internal_error",
				Code:    "internal_error",
			},
		})
	}
	modelParse, err := h.converter.ParseModelName(request.Model)
	if err != nil {
		log.Debugf("Model parsing failed: %v", err)
		return c.Status(400).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: fmt.Sprintf("Invalid model name: %s", request.Model),
				Type:    "invalid_request_error",
				Code:    "invalid_model",
				Param:   "model",
			},
		})
	}

	log.Debugf("Model parse result: Backend=%s, OriginalName=%s, DisplayName=%s",
		modelParse.Backend, modelParse.OriginalName, modelParse.DisplayName)

	// Convert Ollama request to OpenAI format
	log.Debug("Converting request to OpenAI format")

	// Add nil check for Options before accessing fields
	var temperature, topP *float64
	var maxTokens *int
	if request.Options != nil {
		temperature = request.Options.Temperature
		topP = request.Options.TopP
		maxTokens = request.Options.NumPredict
		log.Debugf("Options found - Temperature: %v, TopP: %v, NumPredict: %v",
			temperature, topP, maxTokens)
	} else {
		log.Debug("No options provided, using defaults")
	}

	openAIRequest := &models.OpenAIChatCompletionRequest{
		Model:       modelParse.OriginalName, // Use original name for backend
		Messages:    make([]models.OpenAIMessage, len(request.Messages)),
		Stream:      request.Stream,
		Temperature: temperature,
		TopP:        topP,
		MaxTokens:   maxTokens,
	}

	log.Debugf("Converting %d messages", len(request.Messages))
	for i, msg := range request.Messages {
		log.Debugf("Processing message %d: Role=%s, Content=%v",
			i, msg.Role, msg.Content)
		openAIRequest.Messages[i] = models.OllamaToOpenAIMessage(msg)
		log.Debugf("Converted message %d successfully", i)
	}

	// Convert Ollama think parameter to OpenAI thinking format
	if request.Think {
		openAIRequest.Thinking = &models.OpenAIThinking{
			Type: "enabled",
		}
		log.Debug("Think parameter detected, converted to thinking format")
	}

	// Add nil check for Options
	log.Debug("Checking request.Options")
	if request.Options != nil {
		log.Debug("Options is not nil, accessing fields")
		if request.Options.Temperature != nil {
			log.Debugf("Temperature: %f", *request.Options.Temperature)
		}
		if request.Options.TopP != nil {
			log.Debugf("TopP: %f", *request.Options.TopP)
		}
		if request.Options.NumPredict != nil {
			log.Debugf("NumPredict: %d", *request.Options.NumPredict)
		}
	} else {
		log.Debug("WARNING: request.Options is nil")
	}

	// Handle streaming
	if request.Stream {
		log.Debug("Handling streaming request")
		return h.handleStreamingChat(c, backendClient, openAIRequest)
	}

	// Handle non-streaming
	log.Debug("Handling non-streaming request")
	log.Debug("Calling backendClient.ChatCompletion")
	response, err := backendClient.ChatCompletion(openAIRequest)
	if err != nil {
		log.Debugf("ChatCompletion failed: %v", err)
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: fmt.Sprintf("Chat completion failed: %v", err),
				Type:    "api_error",
				Code:    "chat_completion_failed",
			},
		})
	}

	log.Debug("ChatCompletion succeeded, checking response")

	// Add nil checks before accessing response fields
	if response == nil {
		log.Debug("ERROR: response is nil")
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Backend returned nil response",
				Type:    "api_error",
				Code:    "nil_response",
			},
		})
	}

	log.Debug("Response is not nil, checking Choices")
	if len(response.Choices) == 0 {
		log.Debug("ERROR: response.Choices is empty")
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Backend returned empty choices",
				Type:    "api_error",
				Code:    "empty_choices",
			},
		})
	}

	log.Debugf("Choices length: %d", len(response.Choices))

	// Check the first choice for validity
	firstChoice := response.Choices[0]
	contentStr, ok := firstChoice.Message.Content.(string)
	if (!ok || len(contentStr) == 0) && firstChoice.Message.Content == nil {
		log.Debug("ERROR: response.Choices[0] has empty or nil content")
		return c.Status(500).JSON(models.ErrorResponse{
			Error: models.APIError{
				Message: "Backend returned choice with empty content",
				Type:    "api_error",
				Code:    "empty_content",
			},
		})
	}

	log.Debug("First choice is valid, checking Message")
	if response.Choices[0].Message.Content == nil {
		log.Debug("WARNING: response.Choices[0].Message.Content is nil")
	}

	// Convert OpenAI response to Ollama format
	log.Debug("Converting response to Ollama format")
	ollamaResponse := &models.OllamaChatResponse{
		Model:     request.Model,
		CreatedAt: time.Now(),
		Message:   models.OpenAIToOllamaMessage(response.Choices[0].Message),
		Done:      true,
	}

	log.Debug("Response conversion completed")

	// Add usage information if available
	if response.Usage != nil {
		log.Debugf("Adding usage info: PromptTokens=%d, CompletionTokens=%d",
			response.Usage.PromptTokens, response.Usage.CompletionTokens)
		ollamaResponse.PromptEvalCount = response.Usage.PromptTokens
		ollamaResponse.EvalCount = response.Usage.CompletionTokens
	} else {
		log.Debug("No usage information available")
	}

	log.Debug("Returning JSON response")
	return c.JSON(ollamaResponse)
}

func (h *OllamaHandler) handleStreamingChat(c *fiber.Ctx, backendClient models.BackendClient, request *models.OpenAIChatCompletionRequest) error {
	log.Debug("Streaming chat handler started")

	// Set SSE headers
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")

	log.Debug("SSE headers set, starting stream")

	// Start streaming
	ch, err := backendClient.StreamChatCompletion(request)
	if err != nil {
		log.Debugf("StreamChatCompletion failed: %v", err)
		return c.Status(500).JSON(models.OllamaErrorResponse{
			Error: fmt.Sprintf("Failed to start stream: %v", err),
		})
	}

	log.Debug("Stream started successfully, reading from channel")

	// Stream responses
	for response := range ch {
		log.Debug("Received streaming response")

		// Add checks for streaming response
		if len(response.Choices) == 0 {
			log.Debug("WARNING: Received streaming response with no choices, skipping")
			continue
		}

		// Check if Delta exists and has content
		if response.Choices[0].Delta == nil {
			log.Debug("WARNING: Received streaming response with nil delta, skipping")
			continue
		}

		log.Debug("Converting streaming response to Ollama format")

		// Convert to Ollama format
		ollamaResponse := &models.OllamaChatResponse{
			Model:     request.Model,
			CreatedAt: time.Now(),
			Message:   models.OpenAIToOllamaMessage(*response.Choices[0].Delta),
			Done:      false,
		}

		// Send as SSE
		data := fmt.Sprintf("data: %s\n\n", mustMarshalJSON(ollamaResponse))
		c.WriteString(data)

		if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
			log.Debugf("Streaming finished with reason: %s", response.Choices[0].FinishReason)
			// Send final chunk
			finalResponse := &models.OllamaChatResponse{
				Model:     request.Model,
				CreatedAt: time.Now(),
				Done:      true,
			}
			finalData := fmt.Sprintf("data: %s\n\n", mustMarshalJSON(finalResponse))
			c.WriteString(finalData)
			break
		}
	}

	// Send done signal
	log.Debug("Sending streaming done signal")
	c.WriteString("data: [DONE]\n\n")

	return nil
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
	ch, err := backendClient.StreamChatCompletion(request)

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
				c.WriteString(data)
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
			c.WriteString(finalData)
			break
		}
	}

	// Send done signal
	c.WriteString("data: [DONE]\n\n")

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

		parameters = fmt.Sprintf(`num_keep                       24
stop                           "<|start_header_id|>"
stop                           "<|end_header_id|>"
stop                           "<|eot_id|>"
num_ctx                        4096`)

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

		parameters = fmt.Sprintf(`num_ctx                        4096
stop                           "<|im_start|>"
stop                           "<|im_end|>"
stop                           "<|im_sep|>"`)

	default:
		modelfile = fmt.Sprintf(`# Modelfile generated by "ollama show"
FROM %s
TEMPLATE """{{ if .System }}System: {{ .System }}

{{ end }}User: {{ .Prompt }}
Assistant: """
PARAMETER num_ctx 4096
`, modelParse.OriginalName)

		template = "{{ if .System }}System: {{ .System }}\n\n{{ end }}User: {{ .Prompt }}\nAssistant: "

		parameters = fmt.Sprintf(`num_ctx                        4096
stop                           "User:"
stop                           "Assistant:"`)
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

// Helper function to estimate parameter count based on model name
func (h *OllamaHandler) estimateParameterCount(modelName string) int64 {
	modelName = strings.ToLower(modelName)

	if strings.Contains(modelName, "70b") || strings.Contains(modelName, "65b") {
		return 8030261248 // ~70B parameters
	} else if strings.Contains(modelName, "34b") || strings.Contains(modelName, "33b") {
		return 3420000000 // ~34B parameters
	} else if strings.Contains(modelName, "13b") {
		return 1300000000 // ~13B parameters
	} else if strings.Contains(modelName, "7b") || strings.Contains(modelName, "8b") {
		return 8030261248 // ~8B parameters
	} else if strings.Contains(modelName, "gpt-4") {
		return 17600000000 // ~17.6B for GPT-4
	} else if strings.Contains(modelName, "gpt-3.5") {
		return 6000000000 // ~6B for GPT-3.5
	} else {
		return 1300000000 // Default ~1.3B
	}
}

// Helper function
func mustMarshalJSON(v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return data
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
	return "", fmt.Errorf("Prompt cannot be empty and no default prompt configured for backend")
}

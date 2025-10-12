package services

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"strings"
	"time"

	"github.com/gemone/oai2ollama/internal/config"
	"github.com/gemone/oai2ollama/internal/models"
)

type ModelConverter struct {
	config *config.Config
}

func NewModelConverter(cfg *config.Config) *ModelConverter {
	return &ModelConverter{config: cfg}
}

func (c *ModelConverter) ConvertOpenAIChatRequest(request *models.OpenAIChatCompletionRequest) (*models.OllamaChatRequest, error) {
	// Parse model name to find backend and original model name
	modelParse, err := c.ParseModelName(request.Model)
	if err != nil {
		return nil, fmt.Errorf("failed to parse model name '%s': %w", request.Model, err)
	}

	// Convert messages
	ollamaMessages := make([]models.OllamaMessage, len(request.Messages))
	for i, msg := range request.Messages {
		ollamaMessages[i] = models.OpenAIToOllamaMessage(msg)
	}

	// Convert options
	options := models.OpenAIToOllamaOptions(request)

	return &models.OllamaChatRequest{
		Model:    modelParse.OriginalName,
		Messages: ollamaMessages,
		Stream:   request.Stream,
		Options:  options,
	}, nil
}

func (c *ModelConverter) ConvertOllamaChatResponse(response *models.OllamaChatResponse, originalModel string) *models.OpenAIChatCompletionResponse {
	// Convert message
	message := models.OllamaToOpenAIMessage(response.Message)

	// Generate OpenAI format response
	openAIResponse := &models.OpenAIChatCompletionResponse{
		ID:      generateChatID(),
		Object:  "chat.completion",
		Created: response.CreatedAt.Unix(),
		Model:   originalModel,
		Choices: []models.OpenAIChatCompletionChoice{
			{
				Index:        0,
				Message:      message,
				FinishReason: "stop",
			},
		},
	}

	// Add usage information if available
	if response.PromptEvalCount > 0 || response.EvalCount > 0 {
		openAIResponse.Usage = &models.OpenAIUsage{
			PromptTokens:     response.PromptEvalCount,
			CompletionTokens: response.EvalCount,
			TotalTokens:      response.PromptEvalCount + response.EvalCount,
		}
	}

	return openAIResponse
}

func (c *ModelConverter) ConvertOllamaChatStreamChunk(response *models.OllamaChatResponse, originalModel string) *models.OpenAIChatCompletionResponse {
	// Convert message delta
	delta := models.OllamaToOpenAIMessage(response.Message)

	return &models.OpenAIChatCompletionResponse{
		ID:      generateChatID(),
		Object:  "chat.completion.chunk",
		Created: response.CreatedAt.Unix(),
		Model:   originalModel,
		Choices: []models.OpenAIChatCompletionChoice{
			{
				Index: 0,
				Delta: &delta,
			},
		},
	}
}

func (c *ModelConverter) ConvertOpenAIModelsList(openaiModels []models.OpenAIModel) []models.OllamaModelInfo {
	var ollamaModels []models.OllamaModelInfo

	for _, openaiModel := range openaiModels {
		// Parse model name to determine backend
		modelParse, err := c.ParseModelName(openaiModel.ID)
		if err != nil {
			// If parsing fails, skip this model
			continue
		}

		// Generate a deterministic digest based on model ID
		digest := c.GenerateModelDigest(openaiModel.ID, openaiModel.Created)

		// Estimate model size based on model name patterns
		size := c.EstimateModelSize(openaiModel.ID)

		// Extract parameter size from model name
		parameterSize := c.ExtractParameterSize(openaiModel.ID)

		// Determine quantization level (default for OpenAI models)
		quantizationLevel := "unknown"

		ollamaModel := models.OllamaModelInfo{
			Name:       openaiModel.ID,
			Model:      openaiModel.ID,
			ModifiedAt: time.Unix(openaiModel.Created, 0),
			Size:       size,
			Digest:     digest,
			Details: map[string]interface{}{
				"parent_model":       "",
				"format":             "gguf",
				"family":             c.DetermineModelFamily(openaiModel.ID),
				"families":           []string{c.DetermineModelFamily(openaiModel.ID)},
				"parameter_size":     parameterSize,
				"quantization_level": quantizationLevel,
				"backend":            modelParse.Backend,
				"original_name":      modelParse.OriginalName,
				"openai_object":      openaiModel.Object,
				"openai_created":     openaiModel.Created,
				"openai_owned_by":    openaiModel.OwnedBy,
			},
		}

		// Add capabilities based on model
		capabilities := c.getModelCapabilities(openaiModel.ID, modelParse.Backend)
		if len(capabilities) > 0 {
			ollamaModel.Details["capabilities"] = capabilities
		}

		ollamaModels = append(ollamaModels, ollamaModel)
	}

	return ollamaModels
}

func (c *ModelConverter) ParseModelName(modelName string) (*models.ModelParseResult, error) {
	// 1. Check exact matches in manual model configurations
	for _, model := range c.config.Models {
		if model.Name == modelName && model.Enabled {
			return &models.ModelParseResult{
				Backend:      model.Backend,
				OriginalName: model.OriginalName,
				DisplayName:  model.DisplayName,
				ExactMatch:   true,
			}, nil
		}
	}

	// 2. Try to parse prefixed models
	for _, backend := range c.config.Backends {
		if !backend.Enabled {
			continue
		}

		if backend.ModelPrefix != nil && backend.ModelPrefix.Enabled {
			prefix := backend.ModelPrefix.Prefix
			if prefix == "" {
				prefix = backend.Name
			}

			separator := backend.ModelPrefix.Separator
			if separator == "" {
				separator = "/"
			}

			expectedPrefix := prefix + separator
			if strings.HasPrefix(modelName, expectedPrefix) {
				originalName := strings.TrimPrefix(modelName, expectedPrefix)
				return &models.ModelParseResult{
					Backend:      backend.Name,
					OriginalName: originalName,
					DisplayName:  modelName,
					Prefixed:     true,
					Prefix:       prefix,
				}, nil
			}
		}
	}

	// 3. Try to find in any backend without prefix
	for _, backend := range c.config.Backends {
		if !backend.Enabled {
			continue
		}

		// This would require checking with the backend if the model exists
		// For now, assume it could be a valid unprefixed model
		if backend.ModelPrefix == nil || !backend.ModelPrefix.Enabled {
			return &models.ModelParseResult{
				Backend:      backend.Name,
				OriginalName: modelName,
				DisplayName:  modelName,
				Prefixed:     false,
			}, nil
		}
	}

	return nil, fmt.Errorf("model not found: %s", modelName)
}

func (c *ModelConverter) getModelCapabilities(modelID, backend string) []string {
	// Default capabilities for all models
	capabilities := []string{"completion"}

	// Add capabilities based on model name patterns
	if strings.Contains(strings.ToLower(modelID), "gpt-4") ||
		strings.Contains(strings.ToLower(modelID), "claude") ||
		strings.Contains(strings.ToLower(modelID), "llama") {
		capabilities = append(capabilities, "tools")
	}

	if strings.Contains(strings.ToLower(modelID), "vision") ||
		strings.Contains(strings.ToLower(modelID), "claude-3") ||
		strings.Contains(strings.ToLower(modelID), "gpt-4-vision") {
		capabilities = append(capabilities, "vision")
	}

	if strings.Contains(strings.ToLower(modelID), "embedding") {
		return []string{"embedding"}
	}

	if strings.Contains(strings.ToLower(modelID), "thinking") ||
		strings.Contains(strings.ToLower(modelID), "o1") {
		capabilities = append(capabilities, "thinking")
	}

	// Check configured model capabilities
	for _, model := range c.config.Models {
		if (model.Name == modelID || (model.OriginalName != "" && model.OriginalName == modelID)) && model.Backend == backend {
			if len(model.Capabilities) > 0 {
				return model.Capabilities
			}
		}
	}

	return capabilities
}

func (c *ModelConverter) GetBackendForModel(modelName string) (*config.BackendConfig, error) {
	parse, err := c.ParseModelName(modelName)
	if err != nil {
		return nil, err
	}

	backend, exists := config.GetBackend(parse.Backend)
	if !exists {
		return nil, fmt.Errorf("backend not found: %s", parse.Backend)
	}

	return backend, nil
}

// Helper functions
func generateChatID() string {
	return fmt.Sprintf("chatcmpl-%d", generateRandomID())
}

func generateRandomID() int64 {
	var b [8]byte
	_, err := rand.Read(b[:])
	if err != nil {
		// fallback to timestamp if random fails
		return time.Now().UnixNano()
	}
	return int64(binary.LittleEndian.Uint64(b[:]))
}

// Helper methods for ModelConverter
func (c *ModelConverter) GenerateModelDigest(modelID string, created int64) string {
	// Generate a SHA256 digest based on model ID and creation time
	data := fmt.Sprintf("%s-%d", modelID, created)
	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("%x", hash)
}

func (c *ModelConverter) EstimateModelSize(modelID string) int64 {
	// Estimate model size based on model name patterns
	// These are rough estimates for demonstration
	modelID = strings.ToLower(modelID)

	switch {
	case strings.Contains(modelID, "gpt-4"):
		if strings.Contains(modelID, "32k") {
			return 78 * 1024 * 1024 * 1024 // 78GB for 32K context
		}
		return 38 * 1024 * 1024 * 1024 // 38GB for standard GPT-4
	case strings.Contains(modelID, "gpt-3.5"):
		return 6 * 1024 * 1024 * 1024 // 6GB for GPT-3.5
	case strings.Contains(modelID, "claude-3"):
		if strings.Contains(modelID, "opus") {
			return 30 * 1024 * 1024 * 1024 // 30GB for Claude-3 Opus
		} else if strings.Contains(modelID, "sonnet") {
			return 15 * 1024 * 1024 * 1024 // 15GB for Claude-3 Sonnet
		}
		return 10 * 1024 * 1024 * 1024 // 10GB for Claude-3 Haiku
	case strings.Contains(modelID, "llama"):
		if strings.Contains(modelID, "70b") || strings.Contains(modelID, "65b") {
			return 140 * 1024 * 1024 * 1024 // 140GB for 70B models
		} else if strings.Contains(modelID, "34b") || strings.Contains(modelID, "33b") {
			return 65 * 1024 * 1024 * 1024 // 65GB for 34B models
		} else if strings.Contains(modelID, "13b") {
			return 26 * 1024 * 1024 * 1024 // 26GB for 13B models
		} else if strings.Contains(modelID, "7b") || strings.Contains(modelID, "8b") {
			return 14 * 1024 * 1024 * 1024 // 14GB for 7B/8B models
		}
		return 10 * 1024 * 1024 * 1024 // Default estimate
	case strings.Contains(modelID, "glm"):
		if strings.Contains(modelID, "4.6") {
			return 10 * 1024 * 1024 * 1024 // 10GB estimate for GLM-4.6
		} else if strings.Contains(modelID, "4") {
			return 8 * 1024 * 1024 * 1024 // 8GB estimate for GLM-4
		}
		return 6 * 1024 * 1024 * 1024 // Default for GLM models
	default:
		return 4 * 1024 * 1024 * 1024 // 4GB default estimate
	}
}

func (c *ModelConverter) ExtractParameterSize(modelID string) string {
	// Extract parameter size from model name
	modelID = strings.ToLower(modelID)

	switch {
	case strings.Contains(modelID, "70b") || strings.Contains(modelID, "65b"):
		return "70B"
	case strings.Contains(modelID, "34b") || strings.Contains(modelID, "33b"):
		return "34B"
	case strings.Contains(modelID, "13b"):
		return "13B"
	case strings.Contains(modelID, "7b") || strings.Contains(modelID, "8b"):
		return "7B"
	case strings.Contains(modelID, "gpt-4"):
		return "unknown"
	case strings.Contains(modelID, "gpt-3.5"):
		return "unknown"
	case strings.Contains(modelID, "claude"):
		return "unknown"
	case strings.Contains(modelID, "glm-4.6"):
		return "unknown"
	default:
		return "unknown"
	}
}

func (c *ModelConverter) DetermineModelFamily(modelID string) string {
	// Determine model family based on model name
	modelID = strings.ToLower(modelID)

	switch {
	case strings.Contains(modelID, "gpt-4"):
		return "gpt4"
	case strings.Contains(modelID, "gpt-3.5"):
		return "gpt3.5"
	case strings.Contains(modelID, "claude"):
		return "claude"
	case strings.Contains(modelID, "llama"):
		return "llama"
	case strings.Contains(modelID, "glm"):
		return "glm"
	case strings.Contains(modelID, "mistral"):
		return "mistral"
	case strings.Contains(modelID, "codellama"):
		return "codellama"
	case strings.Contains(modelID, "qwen"):
		return "qwen"
	default:
		return "unknown"
	}
}

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
	config          *config.Config
	patternMatcher  *ModelPatternMatcher
	parseCache      *ModelParseCache
	capabilityCache *ModelCapabilityCache
}

func NewModelConverter(cfg *config.Config) *ModelConverter {
	return &ModelConverter{
		config:          cfg,
		patternMatcher:  GetGlobalPatternMatcher(),
		parseCache:      NewModelParseCache(),
		capabilityCache: NewModelCapabilityCache(),
	}
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
	// 首先检查缓存
	if result, found := c.parseCache.Get(modelName); found {
		return result, nil
	}

	// Check if config is nil
	if c == nil || c.config == nil {
		return nil, fmt.Errorf("model converter or config is nil")
	}

	var result *models.ModelParseResult

	// 1. Check exact matches in manual model configurations
	for _, model := range c.config.Models {
		if model.Name == modelName && model.Enabled {
			result = &models.ModelParseResult{
				Backend:      model.Backend,
				OriginalName: model.OriginalName,
				DisplayName:  model.DisplayName,
				ExactMatch:   true,
			}
			break
		}
	}

	// 2. Try to parse prefixed models
	if result == nil {
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
					result = &models.ModelParseResult{
						Backend:      backend.Name,
						OriginalName: originalName,
						DisplayName:  modelName,
						Prefixed:     true,
						Prefix:       prefix,
					}
					break
				}
			}
		}
	}

	// 3. Try to find in any backend without prefix
	if result == nil {
		for _, backend := range c.config.Backends {
			if !backend.Enabled {
				continue
			}

			// This would require checking with the backend if the model exists
			// For now, assume it could be a valid unprefixed model
			if backend.ModelPrefix == nil || !backend.ModelPrefix.Enabled {
				result = &models.ModelParseResult{
					Backend:      backend.Name,
					OriginalName: modelName,
					DisplayName:  modelName,
					Prefixed:     false,
				}
				break
			}
		}
	}

	if result == nil {
		return nil, fmt.Errorf("model not found: %s", modelName)
	}

	// 缓存结果（TTL 30分钟）
	c.parseCache.Set(modelName, result, 30*time.Minute)

	return result, nil
}

func (c *ModelConverter) getModelCapabilities(modelID, backend string) []string {
	// 首先检查缓存
	cacheKey := fmt.Sprintf("%s:%s", modelID, backend)
	if caps, found := c.capabilityCache.Get(cacheKey); found {
		return caps
	}

	// 使用模式匹配器获取基础能力
	matchResult := c.patternMatcher.MatchModel(modelID)
	capabilities := make([]string, len(matchResult.Capabilities))
	copy(capabilities, matchResult.Capabilities)

	// 检查配置的模型能力（优先级更高）
	for _, model := range c.config.Models {
		if (model.Name == modelID || (model.OriginalName != "" && model.OriginalName == modelID)) && model.Backend == backend {
			if len(model.Capabilities) > 0 {
				capabilities = make([]string, len(model.Capabilities))
				copy(capabilities, model.Capabilities)
				break
			}
		}
	}

	// 缓存结果（TTL 1小时）
	c.capabilityCache.Set(cacheKey, capabilities, time.Hour)

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
	// 使用模式匹配器获取大小估算
	matchResult := c.patternMatcher.MatchModel(modelID)
	return matchResult.SizeHint
}

func (c *ModelConverter) ExtractParameterSize(modelID string) string {
	// 使用模式匹配器获取参数大小
	matchResult := c.patternMatcher.MatchModel(modelID)
	return matchResult.ParamSize
}

func (c *ModelConverter) DetermineModelFamily(modelID string) string {
	// 使用模式匹配器获取模型家族
	matchResult := c.patternMatcher.MatchModel(modelID)
	return matchResult.Family
}

package models

import (
	"fmt"
	"time"
)

// OpenAI API Models

type OpenAIChatCompletionRequest struct {
	Model            string                `json:"model"`
	Messages         []OpenAIMessage       `json:"messages"`
	MaxTokens        *int                  `json:"max_tokens,omitempty"`
	Temperature      *float64              `json:"temperature,omitempty"`
	TopP             *float64              `json:"top_p,omitempty"`
	Stream           bool                  `json:"stream,omitempty"`
	Stop             interface{}           `json:"stop,omitempty"`
	PresencePenalty  *float64              `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64              `json:"frequency_penalty,omitempty"`
	Functions        []OpenAIFunction      `json:"functions,omitempty"`
	FunctionCall     interface{}           `json:"function_call,omitempty"`
	ResponseFormat   *OpenAIResponseFormat `json:"response_format,omitempty"`
	Seed             *int                  `json:"seed,omitempty"`
	Tools            []OpenAITool          `json:"tools,omitempty"`
	ToolChoice       interface{}           `json:"tool_choice,omitempty"`
	User             string                `json:"user,omitempty"`
	Thinking         *OpenAIThinking       `json:"thinking,omitempty"`
}

type OpenAIThinking struct {
	Type string `json:"type"`
}

type OpenAIMessage struct {
	Role         string              `json:"role"`
	Content      interface{}         `json:"content"`
	Name         *string             `json:"name,omitempty"`
	FunctionCall *OpenAIFunctionCall `json:"function_call,omitempty"`
	ToolCalls    []OpenAIToolCall    `json:"tool_calls,omitempty"`
}

type OpenAIFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type OpenAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

type OpenAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function OpenAIFunctionCall `json:"function"`
}

type OpenAIResponseFormat struct {
	Type       string                 `json:"type"`
	JSONSchema map[string]interface{} `json:"json_schema,omitempty"`
}

type OpenAIChatCompletionResponse struct {
	ID                string                       `json:"id"`
	Object            string                       `json:"object"`
	Created           int64                        `json:"created"`
	Model             string                       `json:"model"`
	Choices           []OpenAIChatCompletionChoice `json:"choices"`
	Usage             *OpenAIUsage                 `json:"usage,omitempty"`
	SystemFingerprint *string                      `json:"system_fingerprint,omitempty"`
}

type OpenAIChatCompletionChoice struct {
	Index        int            `json:"index"`
	Message      OpenAIMessage  `json:"message"`
	FinishReason string         `json:"finish_reason"`
	Delta        *OpenAIMessage `json:"delta,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type OpenAIModelsResponse struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// Ollama API Models

type OllamaChatRequest struct {
	Model    string                   `json:"model"`
	Messages []OllamaMessage          `json:"messages"`
	Stream   bool                     `json:"stream,omitempty"`
	Format   string                   `json:"format,omitempty"`
	Options  *OllamaGenerationOptions `json:"options,omitempty"`
	Template string                   `json:"template,omitempty"`
	System   string                   `json:"system,omitempty"`
	Context  []int                    `json:"context,omitempty"`
	Think    bool                     `json:"think,omitempty"`
}

type OllamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaGenerationOptions struct {
	Seed             *int     `json:"seed,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	TopK             *int     `json:"top_k,omitempty"`
	NumPredict       *int     `json:"num_predict,omitempty"`
	NumCtx           *int     `json:"num_ctx,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
}

type OllamaChatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          time.Time     `json:"created_at"`
	Message            OllamaMessage `json:"message"`
	Done               bool          `json:"done"`
	TotalDuration      int64         `json:"total_duration,omitempty"`
	LoadDuration       int64         `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64         `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       int64         `json:"eval_duration,omitempty"`
}

type OllamaModelInfo struct {
	Name       string                 `json:"name"`
	Model      string                 `json:"model"`
	ModifiedAt time.Time              `json:"modified_at"`
	Size       int64                  `json:"size"`
	Digest     string                 `json:"digest"`
	Details    map[string]interface{} `json:"details"`
	ExpiresAt  *time.Time             `json:"expires_at,omitempty"`
	SizeVRAM   int64                  `json:"size_vram,omitempty"`
}

type OllamaModelsResponse struct {
	Models []OllamaModelInfo `json:"models"`
}

type OllamaGenerateRequest struct {
	Model    string                   `json:"model"`
	Prompt   string                   `json:"prompt"`
	Stream   bool                     `json:"stream,omitempty"`
	Format   string                   `json:"format,omitempty"`
	Options  *OllamaGenerationOptions `json:"options,omitempty"`
	Template string                   `json:"template,omitempty"`
	System   string                   `json:"system,omitempty"`
	Context  []int                    `json:"context,omitempty"`
	Raw      bool                     `json:"raw,omitempty"`
	Images   []string                 `json:"images,omitempty"`
	Think    bool                     `json:"think,omitempty"`
}

type OllamaGenerateResponse struct {
	Model              string    `json:"model"`
	CreatedAt          time.Time `json:"created_at"`
	Response           string    `json:"response,omitempty"`
	Done               bool      `json:"done"`
	Context            []int     `json:"context,omitempty"`
	TotalDuration      int64     `json:"total_duration,omitempty"`
	LoadDuration       int64     `json:"load_duration,omitempty"`
	PromptEvalCount    int       `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64     `json:"prompt_eval_duration,omitempty"`
	EvalCount          int       `json:"eval_count,omitempty"`
	EvalDuration       int64     `json:"eval_duration,omitempty"`
}

type OllamaEmbeddingsRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Options map[string]interface{} `json:"options,omitempty"`
}

type OllamaEmbeddingsResponse struct {
	Embedding []float64 `json:"embedding"`
}

type OllamaShowRequest struct {
	Name    string `json:"name"`
	Model   string `json:"model"`
	Verbose bool   `json:"verbose,omitempty"`
}

type OllamaShowResponse struct {
	Modelfile    string                 `json:"modelfile"`
	Parameters   string                 `json:"parameters"`
	Template     string                 `json:"template"`
	Details      map[string]interface{} `json:"details"`
	ModelInfo    map[string]interface{} `json:"model_info,omitempty"`
	ModifiedAt   time.Time              `json:"modified_at"`
	Capabilities []string               `json:"capabilities,omitempty"`
}

// Internal Models

type ModelParseResult struct {
	Backend      string `json:"backend"`
	OriginalName string `json:"original_name"`
	DisplayName  string `json:"display_name"`
	Prefixed     bool   `json:"prefixed"`
	Prefix       string `json:"prefix,omitempty"`
	ExactMatch   bool   `json:"exact_match"`
}

type BackendClient interface {
	GetModels() ([]OpenAIModel, error)
	ChatCompletion(request *OpenAIChatCompletionRequest) (*OpenAIChatCompletionResponse, error)
	StreamChatCompletion(request *OpenAIChatCompletionRequest) (<-chan OpenAIChatCompletionResponse, error)
	GenerateEmbeddings(request *OllamaEmbeddingsRequest) (*OllamaEmbeddingsResponse, error)
}

// Error Models

type APIError struct {
	Message   string `json:"message"`
	Type      string `json:"type"`
	Code      string `json:"code"`
	Param     string `json:"param,omitempty"`
	Timestamp string `json:"timestamp,omitempty"`
	RequestID string `json:"request_id,omitempty"`
}

type ErrorResponse struct {
	Error APIError `json:"error"`
}

// Simple error response for Ollama CLI compatibility
type OllamaErrorResponse struct {
	Error string `json:"error"`
}

// Utility functions for converting between formats

func OpenAIToOllamaMessage(msg OpenAIMessage) OllamaMessage {
	var content string
	switch v := msg.Content.(type) {
	case string:
		content = v
	default:
		// Convert to JSON string for complex content
		content = fmt.Sprintf("%v", v)
	}

	return OllamaMessage{
		Role:    msg.Role,
		Content: content,
	}
}

func OllamaToOpenAIMessage(msg OllamaMessage) OpenAIMessage {
	return OpenAIMessage{
		Role:    msg.Role,
		Content: msg.Content,
	}
}

func OpenAIToOllamaOptions(request *OpenAIChatCompletionRequest) *OllamaGenerationOptions {
	options := &OllamaGenerationOptions{}

	if request.Temperature != nil {
		options.Temperature = request.Temperature
	}
	if request.TopP != nil {
		options.TopP = request.TopP
	}
	if request.MaxTokens != nil {
		options.NumPredict = request.MaxTokens
	}
	if request.PresencePenalty != nil {
		options.PresencePenalty = request.PresencePenalty
	}
	if request.FrequencyPenalty != nil {
		options.FrequencyPenalty = request.FrequencyPenalty
	}
	if request.Stop != nil {
		if stop, ok := request.Stop.(string); ok {
			options.Stop = []string{stop}
		} else if stops, ok := request.Stop.([]string); ok {
			options.Stop = stops
		}
	}

	return options
}

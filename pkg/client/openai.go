package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gemone/oai2ollama/internal/config"
	"github.com/gemone/oai2ollama/internal/models"

	"github.com/gofiber/fiber/v2/log"
)

const (
	sanitizedHeaderPrefixLen = 8
	sanitizedHeaderSuffixLen = 4
)

type OpenAIClient struct {
	config   *config.BackendConfig
	client   *http.Client
	baseURL  string
	modelURL string
	apiKey   string
	debug    bool
}

func NewOpenAIClient(backendConfig *config.BackendConfig) *OpenAIClient {
	// Check if debug mode is enabled globally
	debug := false
	if config.GlobalConfig != nil {
		debug = config.GlobalConfig.Server.Debug
	}

	// Use HTTP client with forced HTTP/1.1 to avoid HTTP/2 issues
	client := &http.Client{
		Timeout: 0, // No timeout for streaming, handle per request
		Transport: &http.Transport{
			ForceAttemptHTTP2:   false, // Force HTTP/1.1 to avoid protocol issues
			MaxIdleConns:        10,
			MaxIdleConnsPerHost: 5,
			IdleConnTimeout:     30 * time.Second,
		},
	}

	return &OpenAIClient{
		config:   backendConfig,
		client:   client,
		baseURL:  backendConfig.BaseURL,
		modelURL: backendConfig.ModelURL,
		apiKey:   backendConfig.APIKey,
		debug:    debug,
	}
}

func (c *OpenAIClient) GetModels() ([]models.OpenAIModel, error) {
	var url string
	if c.modelURL != "" {
		// If modelURL doesn't end with /models, append it
		if !strings.HasSuffix(c.modelURL, "/models") {
			url = fmt.Sprintf("%s/models", c.modelURL)
		} else {
			url = c.modelURL
		}
	} else {
		// Fallback to baseURL + /models
		url = fmt.Sprintf("%s/models", c.baseURL)
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	req.Header.Set("Content-Type", "application/json")

	if c.debug {
		log.Debugf("GetModels Request: GET %s", url)
		log.Debugf("GetModels Headers: %+v", sanitizeHeadersForLogging(req.Header))
	}

	// Use a client with timeout for non-streaming requests
	timeoutClient := c.getTimeoutClient()
	resp, err := timeoutClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get models: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if c.debug {
		log.Debugf("GetModels Response Status: %d", resp.StatusCode)
		log.Debugf("GetModels Response Body: %s", string(body))
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var response models.OpenAIModelsResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if c.debug {
		log.Debugf("GetModels Parsed Response: %+v", response)
	}

	return response.Data, nil
}

func (c *OpenAIClient) ChatCompletion(request *models.OpenAIChatCompletionRequest) (*models.OpenAIChatCompletionResponse, error) {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	// Detect thinking mode
	thinkingEnabled := request.Thinking != nil && request.Thinking.Type == "enabled"

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	if c.debug {
		log.Debugf("ChatCompletion Request: POST %s", url)
		log.Debugf("ChatCompletion Request Body: %s", string(jsonData))
		if thinkingEnabled {
			log.Debug("ChatCompletion: Thinking mode detected (type: enabled)")
		}
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	req.Header.Set("Content-Type", "application/json")

	if c.debug {
		log.Debugf("ChatCompletion Headers: %+v", sanitizeHeadersForLogging(req.Header))
	}

	// Use appropriate client based on thinking mode
	var resp *http.Response
	if thinkingEnabled {
		if c.debug {
			log.Debug("ChatCompletion: Using no-timeout client for thinking mode")
		}
		resp, err = c.client.Do(req) // Use client with no timeout for thinking mode
	} else {
		timeoutClient := c.getTimeoutClient()
		resp, err = timeoutClient.Do(req) // Use client with timeout for normal mode
	}

	if err != nil {
		return nil, fmt.Errorf("failed to complete chat: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if c.debug {
		log.Debugf("ChatCompletion Response Status: %d", resp.StatusCode)
		log.Debugf("ChatCompletion Response Body: %s", string(body))
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var response models.OpenAIChatCompletionResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if c.debug {
		log.Debugf("ChatCompletion Parsed Response: %+v", response)
	}

	return &response, nil
}

func (c *OpenAIClient) StreamChatCompletion(ctx context.Context, request *models.OpenAIChatCompletionRequest) (<-chan models.OpenAIChatCompletionResponse, error) {
	return c.StreamChatCompletionWithChunkTimeout(ctx, request)
}

func (c *OpenAIClient) StreamChatCompletionWithChunkTimeout(ctx context.Context, request *models.OpenAIChatCompletionRequest) (<-chan models.OpenAIChatCompletionResponse, error) {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	// Ensure streaming is enabled without modifying the original request
	reqCopy := *request
	reqCopy.Stream = true

	jsonData, err := json.Marshal(&reqCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Detect thinking mode
	thinkingEnabled := request.Thinking != nil && request.Thinking.Type == "enabled"

	if c.debug {
		log.Debugf("StreamChatCompletion Request: POST %s", url)
		log.Debugf("StreamChatCompletion Request Body: %s", string(jsonData))
		log.Debug("StreamChatCompletion: Chunk timeout recalculation enabled")
		if thinkingEnabled {
			log.Debug("StreamChatCompletion: Thinking mode detected (type: enabled)")
		}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	if c.debug {
		log.Debugf("StreamChatCompletion Headers: %+v", sanitizeHeadersForLogging(req.Header))
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to start stream: %w", err)
	}

	if c.debug {
		log.Debugf("StreamChatCompletion Response Status: %d", resp.StatusCode)
		log.Debugf("StreamChatCompletion Response Headers: %+v", sanitizeHeadersForLogging(resp.Header))
		log.Debug("StreamChatCompletion: Starting to read response body")
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	ch := make(chan models.OpenAIChatCompletionResponse)

	go func() {
		defer resp.Body.Close()
		defer close(ch)

		if c.debug {
			log.Debug("StreamChatCompletion: Starting goroutine to read SSE data")
		}

		reader := bufio.NewReader(resp.Body)

		for {
			select {
			case <-ctx.Done():
				log.Debug("StreamChatCompletion: Context cancelled, closing stream.")
				return
			default:
				line, err := reader.ReadBytes('\n')
				if err != nil {
					if err == io.EOF {
						log.Debug("StreamChatCompletion: EOF reached")
					} else {
						log.Debugf("StreamChatCompletion error reading line: %v", err)
					}
					return
				}

				lineStr := strings.TrimSpace(string(line))
				if c.debug {
					log.Debugf("StreamChatCompletion Raw SSE Line: '%s'", lineStr)
				}

				if lineStr == "" {
					continue
				}

				if c.debug {
					log.Debugf("StreamChatCompletion SSE Line: %s", lineStr)
				}

				if lineStr == "data: [DONE]" {
					if c.debug {
						log.Debug("StreamChatCompletion: Received [DONE]")
					}
					return
				}

				if !strings.HasPrefix(lineStr, "data: ") {
					continue
				}

				jsonData := []byte(strings.TrimPrefix(lineStr, "data: "))

				var response models.OpenAIChatCompletionResponse
				if err := json.Unmarshal(jsonData, &response); err != nil {
					if c.debug {
						log.Debugf("StreamChatCompletion error unmarshaling: %v, data: %s", err, string(jsonData))
					}
					continue
				}

				if c.debug {
					log.Debugf("StreamChatCompletion Parsed Response: %+v", response)
				}

				ch <- response
			}
		}
	}()

	return ch, nil
}

func (c *OpenAIClient) GenerateEmbeddings(request *models.OllamaEmbeddingsRequest) (*models.OllamaEmbeddingsResponse, error) {
	url := fmt.Sprintf("%s/embeddings", c.baseURL)

	// Convert Ollama request to OpenAI format
	openAIRequest := map[string]interface{}{
		"model": request.Model,
		"input": request.Prompt,
	}

	if request.Options != nil {
		openAIRequest["encoding_format"] = "float"
		if dimensions, ok := request.Options["dimensions"].(int); ok {
			openAIRequest["dimensions"] = dimensions
		}
	}

	jsonData, err := json.Marshal(openAIRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	if c.debug {
		log.Debugf("GenerateEmbeddings Request: POST %s", url)
		log.Debugf("GenerateEmbeddings Request Body: %s", string(jsonData))
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	req.Header.Set("Content-Type", "application/json")

	if c.debug {
		log.Debugf("GenerateEmbeddings Headers: %+v", sanitizeHeadersForLogging(req.Header))
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embeddings: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if c.debug {
		log.Debugf("GenerateEmbeddings Response Status: %d", resp.StatusCode)
		log.Debugf("GenerateEmbeddings Response Body: %s", string(body))
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var openAIResponse struct {
		Object string `json:"object"`
		Data   []struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Model string `json:"model"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &openAIResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if c.debug {
		log.Debugf("GenerateEmbeddings Parsed Response: %+v", openAIResponse)
	}

	if len(openAIResponse.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	// Convert to Ollama format
	return &models.OllamaEmbeddingsResponse{
		Embedding: openAIResponse.Data[0].Embedding,
	}, nil
}

// Helper methods for timeout management
func (c *OpenAIClient) getTimeoutClient() *http.Client {
	timeout := time.Duration(c.config.Timeout) * time.Second
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	// Create a simple timeout client with default transport
	return &http.Client{
		Timeout: timeout,
		// Use default transport to avoid HTTP/2 issues
	}
}

func sanitizeHeadersForLogging(headers http.Header) map[string]string {
	sanitized := make(map[string]string)

	for key, values := range headers {
		lowerKey := strings.ToLower(key)
		// Hide sensitive headers
		if lowerKey == "authorization" || lowerKey == "x-api-key" || lowerKey == "cookie" {
			if len(values) > 0 {
				// Show only first few characters to identify the key type
				value := values[0]
				if len(value) > sanitizedHeaderPrefixLen+sanitizedHeaderSuffixLen+3 {
					sanitized[key] = value[:sanitizedHeaderPrefixLen] + "***" + value[len(value)-sanitizedHeaderSuffixLen:]
				} else {
					sanitized[key] = "***"
				}
			} else {
				sanitized[key] = "***"
			}
		} else {
			// Show non-sensitive headers normally
			if len(values) > 0 {
				sanitized[key] = values[0]
			}
		}
	}

	return sanitized
}

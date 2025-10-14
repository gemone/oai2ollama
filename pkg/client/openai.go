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
	"github.com/gemone/oai2ollama/pkg/utils"

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

	// Create optimized HTTP client
	client := utils.NewHTTPClient(backendConfig.BaseURL, 0) // No timeout for streaming, handle per request

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
		// Use baseURL and append /models
		url = fmt.Sprintf("%s/models", c.baseURL)
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response struct {
		Object string               `json:"object"`
		Data   []models.OpenAIModel `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return response.Data, nil
}

func (c *OpenAIClient) ChatCompletion(request *models.OpenAIChatCompletionRequest) (*models.OpenAIChatCompletionResponse, error) {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	jsonData, err := utils.MarshalJSON(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	req.Header.Set("Content-Type", "application/json")

	// Use client with timeout for non-streaming requests
	client := c.client
	if client.Timeout == 0 {
		client = utils.NewHTTPClient(c.baseURL, 60*time.Second)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to complete chat: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response models.OpenAIChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

func (c *OpenAIClient) StreamChatCompletion(ctx context.Context, request *models.OpenAIChatCompletionRequest) (<-chan models.OpenAIChatCompletionResponse, error) {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	// Ensure streaming is enabled without modifying the original request
	reqCopy := *request
	reqCopy.Stream = true

	jsonData, err := utils.MarshalJSON(&reqCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	// Use client without timeout for streaming
	client := c.client
	if client.Timeout != 0 {
		client = utils.NewHTTPClient(c.baseURL, 0)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to start stream: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	ch := make(chan models.OpenAIChatCompletionResponse)

	go func() {
		defer resp.Body.Close()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)

		for {
			select {
			case <-ctx.Done():
				return
			default:
				line, err := reader.ReadBytes('\n')
				if err != nil {
					if err == io.EOF {
						return
					}
					continue
				}

				lineStr := strings.TrimSpace(string(line))
				if lineStr == "" {
					continue
				}

				if lineStr == "data: [DONE]" {
					return
				}

				if !strings.HasPrefix(lineStr, "data: ") {
					continue
				}

				jsonData := []byte(strings.TrimPrefix(lineStr, "data: "))

				var response models.OpenAIChatCompletionResponse
				if err := json.Unmarshal(jsonData, &response); err != nil {
					continue
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

	jsonData, err := utils.MarshalJSON(openAIRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	req.Header.Set("Content-Type", "application/json")

	// Use client with timeout
	client := c.client
	if client.Timeout == 0 {
		client = utils.NewHTTPClient(c.baseURL, 60*time.Second)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embeddings: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var openAIResponse struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&openAIResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(openAIResponse.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	// Convert to Ollama format
	return &models.OllamaEmbeddingsResponse{
		Embedding: openAIResponse.Data[0].Embedding,
	}, nil
}

func (c *OpenAIClient) ProxyRequest(ctx context.Context, req *http.Request, w http.ResponseWriter) error {
	// Create a new request based on the incoming request
	var body io.Reader
	if req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err != nil {
			return fmt.Errorf("failed to read request body: %w", err)
		}
		body = bytes.NewReader(bodyBytes)
	}

	// Build the target URL
	targetURL := c.baseURL
	if req.URL.RawQuery != "" {
		targetURL += "?" + req.URL.RawQuery
	}

	// Create new request
	proxyReq, err := http.NewRequestWithContext(ctx, req.Method, targetURL, body)
	if err != nil {
		return fmt.Errorf("failed to create proxy request: %w", err)
	}

	// Copy headers
	for name, values := range req.Header {
		for _, value := range values {
			proxyReq.Header.Add(name, value)
		}
	}

	// Set API key if configured
	if c.apiKey != "" {
		proxyReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	// Ensure Content-Type is set
	if proxyReq.Header.Get("Content-Type") == "" {
		proxyReq.Header.Set("Content-Type", "application/json")
	}

	// Add OpenAI compatibility headers
	proxyReq.Header.Set("Accept", "application/json, text/event-stream")
	proxyReq.Header.Set("User-Agent", "oai2ollama/1.0")

	// Log the request if debug is enabled
	if c.debug {
		log.Infof("Proxying %s request to: %s", req.Method, targetURL)
		c.logHeaders(proxyReq.Header)
	}

	// Make the request with timeout
	client := c.client
	if client.Timeout == 0 {
		client = utils.NewHTTPClient(c.baseURL, 60*time.Second)
	}

	resp, err := client.Do(proxyReq)
	if err != nil {
		return fmt.Errorf("failed to make proxy request: %w", err)
	}
	defer resp.Body.Close()

	// Copy response headers
	for name, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}

	// Set status code
	w.WriteHeader(resp.StatusCode)

	// Copy response body
	_, err = io.Copy(w, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to copy response body: %w", err)
	}

	return nil
}

func (c *OpenAIClient) ProxyStreamingRequest(ctx context.Context, req *http.Request, w http.ResponseWriter) error {
	// Create a new request based on the incoming request
	var body io.Reader
	if req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err != nil {
			return fmt.Errorf("failed to read request body: %w", err)
		}
		body = bytes.NewReader(bodyBytes)
	}

	// Build the target URL
	targetURL := c.baseURL
	if req.URL.RawQuery != "" {
		targetURL += "?" + req.URL.RawQuery
	}

	// Create new request
	proxyReq, err := http.NewRequestWithContext(ctx, req.Method, targetURL, body)
	if err != nil {
		return fmt.Errorf("failed to create proxy request: %w", err)
	}

	// Copy headers
	for name, values := range req.Header {
		for _, value := range values {
			proxyReq.Header.Add(name, value)
		}
	}

	// Set API key if configured
	if c.apiKey != "" {
		proxyReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	// Ensure Content-Type is set
	if proxyReq.Header.Get("Content-Type") == "" {
		proxyReq.Header.Set("Content-Type", "application/json")
	}

	// Add OpenAI compatibility headers
	proxyReq.Header.Set("Accept", "text/event-stream")
	proxyReq.Header.Set("Cache-Control", "no-cache")
	proxyReq.Header.Set("User-Agent", "oai2ollama/1.0")

	// Log the request if debug is enabled
	if c.debug {
		log.Infof("Proxying streaming %s request to: %s", req.Method, targetURL)
		c.logHeaders(proxyReq.Header)
	}

	// Make the request
	client := c.client
	if client.Timeout == 0 {
		client = utils.NewHTTPClient(c.baseURL, 0) // No timeout for streaming
	}

	resp, err := client.Do(proxyReq)
	if err != nil {
		return fmt.Errorf("failed to make proxy request: %w", err)
	}
	defer resp.Body.Close()

	// Copy response headers
	for name, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}

	// Set status code
	w.WriteHeader(resp.StatusCode)

	// Copy response body with streaming
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if _, err := fmt.Fprint(w, line+"\n"); err != nil {
			return fmt.Errorf("failed to write response: %w", err)
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	return nil
}

func (c *OpenAIClient) logHeaders(headers http.Header) {
	for name, values := range headers {
		for _, value := range values {
			if c.shouldSanitizeHeader(name) {
				log.Infof("Header: %s: %s", name, c.sanitizeHeaderValue(value))
			} else {
				log.Infof("Header: %s: %s", name, value)
			}
		}
	}
}

func (c *OpenAIClient) shouldSanitizeHeader(headerName string) bool {
	headerName = strings.ToLower(headerName)
	sensitiveHeaders := []string{"authorization", "api-key", "x-api-key", "cookie", "set-cookie"}
	for _, sensitive := range sensitiveHeaders {
		if strings.Contains(headerName, sensitive) {
			return true
		}
	}
	return false
}

func (c *OpenAIClient) sanitizeHeaderValue(value string) string {
	if len(value) <= sanitizedHeaderPrefixLen+sanitizedHeaderSuffixLen {
		return strings.Repeat("*", len(value))
	}
	return value[:sanitizedHeaderPrefixLen] + strings.Repeat("*", len(value)-sanitizedHeaderPrefixLen-sanitizedHeaderSuffixLen) + value[len(value)-sanitizedHeaderSuffixLen:]
}

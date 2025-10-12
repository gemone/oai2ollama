package client

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gemone/oai2ollama/internal/config"
	"github.com/gemone/oai2ollama/internal/models"
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
	timeout := time.Duration(backendConfig.Timeout) * time.Second
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	// Check if debug mode is enabled globally
	debug := false
	if config.GlobalConfig != nil {
		debug = config.GlobalConfig.Server.Debug
	}

	return &OpenAIClient{
		config:   backendConfig,
		client:   &http.Client{Timeout: timeout},
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
		log.Printf("[DEBUG] GetModels Request: GET %s", url)
		log.Printf("[DEBUG] GetModels Headers: %+v", req.Header)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get models: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if c.debug {
		log.Printf("[DEBUG] GetModels Response Status: %d", resp.StatusCode)
		log.Printf("[DEBUG] GetModels Response Body: %s", string(body))
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var response models.OpenAIModelsResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if c.debug {
		log.Printf("[DEBUG] GetModels Parsed Response: %+v", response)
	}

	return response.Data, nil
}

func (c *OpenAIClient) ChatCompletion(request *models.OpenAIChatCompletionRequest) (*models.OpenAIChatCompletionResponse, error) {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	if c.debug {
		log.Printf("[DEBUG] ChatCompletion Request: POST %s", url)
		log.Printf("[DEBUG] ChatCompletion Request Body: %s", string(jsonData))
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	req.Header.Set("Content-Type", "application/json")

	if c.debug {
		log.Printf("[DEBUG] ChatCompletion Headers: %+v", req.Header)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to complete chat: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if c.debug {
		log.Printf("[DEBUG] ChatCompletion Response Status: %d", resp.StatusCode)
		log.Printf("[DEBUG] ChatCompletion Response Body: %s", string(body))
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var response models.OpenAIChatCompletionResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if c.debug {
		log.Printf("[DEBUG] ChatCompletion Parsed Response: %+v", response)
	}

	return &response, nil
}

func (c *OpenAIClient) StreamChatCompletion(request *models.OpenAIChatCompletionRequest) (<-chan models.OpenAIChatCompletionResponse, error) {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	// Ensure streaming is enabled without modifying the original request
	reqCopy := *request
	reqCopy.Stream = true

	jsonData, err := json.Marshal(&reqCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	if c.debug {
		log.Printf("[DEBUG] StreamChatCompletion Request: POST %s", url)
		log.Printf("[DEBUG] StreamChatCompletion Request Body: %s", string(jsonData))
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	if c.debug {
		log.Printf("[DEBUG] StreamChatCompletion Headers: %+v", req.Header)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to start stream: %w", err)
	}

	if c.debug {
		log.Printf("[DEBUG] StreamChatCompletion Response Status: %d", resp.StatusCode)
		log.Printf("[DEBUG] StreamChatCompletion Response Headers: %+v", resp.Header)
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

		for {
			var line string
			line, err = readSSELine(resp.Body)
			if err != nil {
				if err == io.EOF {
					if c.debug {
						log.Printf("[DEBUG] StreamChatCompletion: EOF reached")
					}
					return
				}
				if c.debug {
					log.Printf("[DEBUG] StreamChatCompletion error reading line: %v", err)
				}
				continue
			}

			if line == "" {
				continue
			}

			if c.debug {
				log.Printf("[DEBUG] StreamChatCompletion SSE Line: %s", line)
			}

			if line == "data: [DONE]" {
				if c.debug {
					log.Printf("[DEBUG] StreamChatCompletion: Received [DONE]")
				}
				return
			}

			if !bytes.HasPrefix([]byte(line), []byte("data: ")) {
				continue
			}

			jsonData := bytes.TrimPrefix([]byte(line), []byte("data: "))

			var response models.OpenAIChatCompletionResponse
			if err := json.Unmarshal(jsonData, &response); err != nil {
				if c.debug {
					log.Printf("[DEBUG] StreamChatCompletion error unmarshaling: %v, data: %s", err, string(jsonData))
				}
				continue
			}

			if c.debug {
				log.Printf("[DEBUG] StreamChatCompletion Parsed Response: %+v", response)
			}

			ch <- response
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
		log.Printf("[DEBUG] GenerateEmbeddings Request: POST %s", url)
		log.Printf("[DEBUG] GenerateEmbeddings Request Body: %s", string(jsonData))
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	req.Header.Set("Content-Type", "application/json")

	if c.debug {
		log.Printf("[DEBUG] GenerateEmbeddings Headers: %+v", req.Header)
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
		log.Printf("[DEBUG] GenerateEmbeddings Response Status: %d", resp.StatusCode)
		log.Printf("[DEBUG] GenerateEmbeddings Response Body: %s", string(body))
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
		log.Printf("[DEBUG] GenerateEmbeddings Parsed Response: %+v", openAIResponse)
	}

	if len(openAIResponse.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	// Convert to Ollama format
	return &models.OllamaEmbeddingsResponse{
		Embedding: openAIResponse.Data[0].Embedding,
	}, nil
}

// Helper function to read Server-Sent Events lines

func readSSELine(r io.Reader) (string, error) {
	reader := bufio.NewReader(r)
	line, err := reader.ReadString('\n')
	if err != nil {
		return "", err
	}
	// Remove trailing \r and \n
	line = strings.TrimRight(line, "\r\n")
	return line, nil
}

package middleware

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
)

// RequestBodyParser 请求体解析器，用于高效提取模型信息
type RequestBodyParser struct {
	bufferPool sync.Pool
}

// NewRequestBodyParser 创建新的请求体解析器
func NewRequestBodyParser() *RequestBodyParser {
	return &RequestBodyParser{
		bufferPool: sync.Pool{
			New: func() interface{} {
				return new(bytes.Buffer)
			},
		},
	}
}

// ParseModelInfo 从请求体中解析模型信息
func (p *RequestBodyParser) ParseModelInfo(body []byte, path string) (string, string) {
	if len(body) == 0 {
		return "", ""
	}

	// 使用缓冲池避免频繁分配
	buf := p.bufferPool.Get().(*bytes.Buffer)
	defer func() {
		buf.Reset()
		p.bufferPool.Put(buf)
	}()

	buf.Write(body)

	// 根据不同的端点类型解析
	switch path {
	case "/api/chat", "/v1/chat/completions":
		return p.parseChatRequest(buf.Bytes())
	case "/api/generate":
		return p.parseGenerateRequest(buf.Bytes())
	default:
		return "", ""
	}
}

// parseChatRequest 解析聊天请求
func (p *RequestBodyParser) parseChatRequest(body []byte) (string, string) {
	// 创建一个简化的结构体来解析只需要 model 字段
	var req struct {
		Model string `json:"model"`
	}

	// 使用 json.Decoder 只解析需要的字段，避免完整解析
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields() // 只解析已知字段，提升性能

	if err := decoder.Decode(&req); err != nil {
		// 如果简化解析失败，使用字符串匹配作为后备
		return p.extractModelFromString(string(body))
	}

	backend, model := p.extractBackendFromModel(req.Model)
	return backend, model
}

// parseGenerateRequest 解析生成请求
func (p *RequestBodyParser) parseGenerateRequest(body []byte) (string, string) {
	var req struct {
		Model string `json:"model"`
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&req); err != nil {
		return p.extractModelFromString(string(body))
	}

	backend, model := p.extractBackendFromModel(req.Model)
	return backend, model
}

// extractModelFromString 从字符串中提取模型信息（后备方法）
func (p *RequestBodyParser) extractModelFromString(bodyStr string) (string, string) {
	if !strings.Contains(bodyStr, `"model"`) {
		return "", ""
	}

	// 使用更高效的字符串操作
	parts := strings.SplitN(bodyStr, `"model"`, 2)
	if len(parts) < 2 {
		return "", ""
	}

	// 找到模型值
	modelPart := parts[1]
	// 跳过空白和冒号
	modelPart = strings.TrimLeft(modelPart, " \t\n\r:")
	if len(modelPart) == 0 {
		return "", ""
	}

	// 提取引号内的值
	if modelPart[0] == '"' {
		endQuote := strings.IndexByte(modelPart[1:], '"')
		if endQuote == -1 {
			return "", ""
		}
		model := modelPart[1 : endQuote+1]
		backend, _ := p.extractBackendFromModel(model)
		return backend, model
	}

	return "", ""
}

// extractBackendFromModel 从模型名称中提取后端信息
func (p *RequestBodyParser) extractBackendFromModel(model string) (string, string) {
	if model == "" {
		return "", ""
	}

	// 如果包含前缀，提取后端信息
	if strings.Contains(model, "/") {
		parts := strings.SplitN(model, "/", 2)
		if len(parts) == 2 {
			return parts[0], model
		}
	}

	return "", model
}

// TokenUsageExtractor Token使用量提取器
type TokenUsageExtractor struct {
	bufferPool sync.Pool
}

// NewTokenUsageExtractor 创建新的Token使用量提取器
func NewTokenUsageExtractor() *TokenUsageExtractor {
	return &TokenUsageExtractor{
		bufferPool: sync.Pool{
			New: func() interface{} {
				return new(bytes.Buffer)
			},
		},
	}
}

// ExtractTokenUsage 从响应中提取Token使用量
func (e *TokenUsageExtractor) ExtractTokenUsage(responseBody []byte) (promptTokens, totalTokens int) {
	if len(responseBody) == 0 {
		return 0, 0
	}

	// 检查响应头中的Token信息（优先）
	// 这里可以添加从响应头提取Token的逻辑

	// 从响应体中提取Token信息
	buf := e.bufferPool.Get().(*bytes.Buffer)
	defer func() {
		buf.Reset()
		e.bufferPool.Put(buf)
	}()

	buf.Write(responseBody)

	// 尝试解析不同格式的响应
	if prompt, total := e.extractFromOpenAIFormat(buf.Bytes()); prompt > 0 || total > 0 {
		return prompt, total
	}

	if prompt, total := e.extractFromOllamaFormat(buf.Bytes()); prompt > 0 || total > 0 {
		return prompt, total
	}

	return 0, 0
}

// extractFromOpenAIFormat 从OpenAI格式中提取Token使用量
func (e *TokenUsageExtractor) extractFromOpenAIFormat(body []byte) (promptTokens, totalTokens int) {
	var response struct {
		Usage *struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return 0, 0
	}

	if response.Usage != nil {
		return response.Usage.PromptTokens, response.Usage.TotalTokens
	}

	return 0, 0
}

// extractFromOllamaFormat 从Ollama格式中提取Token使用量
func (e *TokenUsageExtractor) extractFromOllamaFormat(body []byte) (promptTokens, totalTokens int) {
	var response struct {
		PromptEvalCount int `json:"prompt_eval_count"`
		EvalCount       int `json:"eval_count"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return 0, 0
	}

	return response.PromptEvalCount, response.PromptEvalCount + response.EvalCount
}

// 全局解析器实例
var globalRequestBodyParser *RequestBodyParser
var globalTokenUsageExtractor *TokenUsageExtractor
var parserOnce sync.Once

// GetGlobalRequestBodyParser 获取全局请求体解析器
func GetGlobalRequestBodyParser() *RequestBodyParser {
	parserOnce.Do(func() {
		globalRequestBodyParser = NewRequestBodyParser()
		globalTokenUsageExtractor = NewTokenUsageExtractor()
	})
	return globalRequestBodyParser
}

// GetGlobalTokenUsageExtractor 获取全局Token使用量提取器
func GetGlobalTokenUsageExtractor() *TokenUsageExtractor {
	return globalTokenUsageExtractor
}

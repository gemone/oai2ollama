package middleware

import (
	"fmt"
	"strings"
	"time"

	"github.com/gemone/oai2ollama/internal/models"
	"github.com/gemone/oai2ollama/internal/services"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/log"
	"github.com/google/uuid"
)

// MetricsMiddleware creates a middleware for collecting API metrics
func MetricsMiddleware(metricsService *services.MetricsService) fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Skip metrics collection if service is disabled
		if metricsService == nil {
			return c.Next()
		}

		start := time.Now()

		// Generate request ID
		requestID := c.Get("X-Request-ID")
		if requestID == "" {
			requestID = uuid.New().String()
		}
		c.Set("X-Request-ID", requestID)

		// Store request body size if needed
		var requestSize int64
		if c.Request().Body() != nil {
			requestSize = int64(len(c.Request().Body()))
		}

		// Continue processing the request
		err := c.Next()

		// Calculate duration
		duration := time.Since(start)

		// Extract metrics from response
		metric := &models.APIMetrics{
			Timestamp:    start,
			Method:       c.Method(),
			Endpoint:     c.Path(),
			Status:       c.Response().StatusCode(),
			Duration:     duration.Milliseconds(),
			RequestSize:  requestSize,
			ResponseSize: int64(len(c.Response().Body())),
			UserAgent:    c.Get("User-Agent"),
			ClientIP:     c.IP(),
			RequestID:    requestID,
		}

		// Extract model and backend information from request
		metric.Model, metric.Backend = extractModelInfo(c)

		// Extract token usage from response if available
		metric.PromptTokens, metric.TotalTokens = extractTokenUsage(c)

		// Determine if this was a streaming request
		metric.Streaming = isStreamingRequest(c)

		// Record error if any
		if err != nil {
			metric.Error = err.Error()
		} else if c.Response().StatusCode() >= 400 {
			metric.Error = fmt.Sprintf("HTTP %d", c.Response().StatusCode())
		}

		// Record the metric asynchronously
		go func() {
			if recordErr := metricsService.RecordMetric(metric); recordErr != nil {
				log.Errorf("Failed to record metric: %v", recordErr)
			}
		}()

		return err
	}
}

// extractModelInfo extracts model and backend information from the request
func extractModelInfo(c *fiber.Ctx) (string, string) {
	var model, backend string

	// Try to extract from different request formats
	switch c.Path() {
	case "/api/chat", "/v1/chat/completions":
		// For chat endpoints, try to parse the request body
		if c.Method() == "POST" {
			body := c.Request().Body()
			if len(body) > 0 {
				// Simple JSON parsing to extract model
				bodyStr := string(body)
				if strings.Contains(bodyStr, `"model"`) {
					// Extract model value (simple approach)
					parts := strings.Split(bodyStr, `"model"`)
					if len(parts) > 1 {
						modelPart := parts[1]
						if idx := strings.Index(modelPart, `"`); idx > 0 {
							modelPart = modelPart[idx+1:]
							if idx := strings.Index(modelPart, `"`); idx > 0 {
								model = modelPart[:idx]
							}
						}
					}
				}
			}
		}
	case "/api/generate":
		// For generate endpoint
		if c.Method() == "POST" {
			body := c.Request().Body()
			if len(body) > 0 {
				bodyStr := string(body)
				if strings.Contains(bodyStr, `"model"`) {
					parts := strings.Split(bodyStr, `"model"`)
					if len(parts) > 1 {
						modelPart := parts[1]
						if idx := strings.Index(modelPart, `"`); idx > 0 {
							modelPart = modelPart[idx+1:]
							if idx := strings.Index(modelPart, `"`); idx > 0 {
								model = modelPart[:idx]
							}
						}
					}
				}
			}
		}
	}

	// Extract backend from model name if model contains prefix
	if model != "" && strings.Contains(model, "/") {
		parts := strings.SplitN(model, "/", 2)
		if len(parts) == 2 {
			backend = parts[0]
			// Keep the full model name including prefix
		}
	}

	return model, backend
}

// extractTokenUsage extracts token usage information from the response
func extractTokenUsage(c *fiber.Ctx) (int, int) {
	var promptTokens, totalTokens int

	// Try to extract from response headers first (use GetRespHeader for response headers)
	if usageHeader := c.GetRespHeader("X-Token-Usage"); usageHeader != "" {
		// Format: "prompt,total" (e.g., "8,251")
		if strings.Contains(usageHeader, ",") {
			parts := strings.Split(usageHeader, ",")
			if len(parts) == 2 {
				// Parse prompt tokens
				if _, err := fmt.Sscanf(strings.TrimSpace(parts[0]), "%d", &promptTokens); err == nil {
					// Parse total tokens
					if _, err := fmt.Sscanf(strings.TrimSpace(parts[1]), "%d", &totalTokens); err == nil {
						return promptTokens, totalTokens
					}
				}
			}
		}
	}

	// Try to extract from response body for JSON responses
	responseBody := c.Response().Body()
	if len(responseBody) > 0 {
		bodyStr := string(responseBody)

		// Look for OpenAI usage format
		if strings.Contains(bodyStr, `"usage"`) {
			// Extract prompt_tokens
			if strings.Contains(bodyStr, `"prompt_tokens"`) {
				parts := strings.Split(bodyStr, `"prompt_tokens"`)
				if len(parts) > 1 {
					tokenPart := parts[1]
					// Find the number after the colon
					if colonIdx := strings.Index(tokenPart, ":"); colonIdx > 0 {
						tokenPart = tokenPart[colonIdx+1:]
						// Extract number until comma or closing brace
						endIdx := strings.IndexAny(tokenPart, ",}")
						if endIdx > 0 {
							tokenPart = tokenPart[:endIdx]
							if _, err := fmt.Sscanf(strings.TrimSpace(tokenPart), "%d", &promptTokens); err == nil {
								// Successfully extracted prompt tokens
							}
						}
					}
				}
			}

			// Extract total_tokens
			if strings.Contains(bodyStr, `"total_tokens"`) {
				parts := strings.Split(bodyStr, `"total_tokens"`)
				if len(parts) > 1 {
					tokenPart := parts[1]
					if colonIdx := strings.Index(tokenPart, ":"); colonIdx > 0 {
						tokenPart = tokenPart[colonIdx+1:]
						endIdx := strings.IndexAny(tokenPart, ",}")
						if endIdx > 0 {
							tokenPart = tokenPart[:endIdx]
							if _, err := fmt.Sscanf(strings.TrimSpace(tokenPart), "%d", &totalTokens); err == nil {
								// Successfully extracted total tokens
							}
						}
					}
				}
			}
		}

		// Look for Ollama format (prompt_eval_count, eval_count)
		if strings.Contains(bodyStr, `"prompt_eval_count"`) {
			parts := strings.Split(bodyStr, `"prompt_eval_count"`)
			if len(parts) > 1 {
				tokenPart := parts[1]
				if colonIdx := strings.Index(tokenPart, ":"); colonIdx > 0 {
					tokenPart = tokenPart[colonIdx+1:]
					endIdx := strings.IndexAny(tokenPart, ",}")
					if endIdx > 0 {
						tokenPart = tokenPart[:endIdx]
						if _, err := fmt.Sscanf(strings.TrimSpace(tokenPart), "%d", &promptTokens); err == nil {
							// Successfully extracted prompt eval count
						}
					}
				}
			}
		}

		if strings.Contains(bodyStr, `"eval_count"`) {
			parts := strings.Split(bodyStr, `"eval_count"`)
			if len(parts) > 1 {
				tokenPart := parts[1]
				if colonIdx := strings.Index(tokenPart, ":"); colonIdx > 0 {
					tokenPart = tokenPart[colonIdx+1:]
					endIdx := strings.IndexAny(tokenPart, ",}")
					if endIdx > 0 {
						tokenPart = tokenPart[:endIdx]
						var evalCount int
						if _, err := fmt.Sscanf(strings.TrimSpace(tokenPart), "%d", &evalCount); err == nil {
							totalTokens = promptTokens + evalCount
						}
					}
				}
			}
		}
	}

	return promptTokens, totalTokens
}

// isStreamingRequest determines if the request was a streaming request
func isStreamingRequest(c *fiber.Ctx) bool {
	// Check request body for stream parameter
	if c.Method() == "POST" {
		body := c.Request().Body()
		if len(body) > 0 {
			bodyStr := string(body)
			return strings.Contains(bodyStr, `"stream":true`) || strings.Contains(bodyStr, `"stream": true`)
		}
	}

	// Check response headers (use GetRespHeader for response headers)
	if contentType := c.GetRespHeader("Content-Type"); strings.Contains(contentType, "text/event-stream") {
		return true
	}

	return false
}

// SetTokenUsage is a helper function to set token usage in response headers
func SetTokenUsage(c *fiber.Ctx, promptTokens, completionTokens, totalTokens int) {
	usageHeader := fmt.Sprintf("%d,%d", promptTokens, totalTokens)
	c.Set("X-Token-Usage", usageHeader)
}

// MetricsContextKey is the key used to store metrics context in the Fiber context
type MetricsContextKey string

const (
	// ContextKeyMetrics is used to store metrics information in the context
	ContextKeyMetrics MetricsContextKey = "metrics"
)

// GetMetricsFromContext retrieves metrics information from the Fiber context
func GetMetricsFromContext(c *fiber.Ctx) *models.APIMetrics {
	if val := c.Locals(string(ContextKeyMetrics)); val != nil {
		if metric, ok := val.(*models.APIMetrics); ok {
			return metric
		}
	}
	return nil
}

// SetMetricsInContext stores metrics information in the Fiber context
func SetMetricsInContext(c *fiber.Ctx, metric *models.APIMetrics) {
	c.Locals(string(ContextKeyMetrics), metric)
}

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
		parser := GetGlobalRequestBodyParser()
		metric.Model, metric.Backend = parser.ParseModelInfo(c.Request().Body(), c.Path())

		// Extract token usage from response if available
		extractor := GetGlobalTokenUsageExtractor()
		metric.PromptTokens, metric.TotalTokens = extractor.ExtractTokenUsage(c.Response().Body())

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

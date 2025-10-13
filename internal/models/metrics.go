package models

import (
	"time"
)

// APIMetrics represents API request metrics
type APIMetrics struct {
	ID           int64     `json:"id" db:"id"`
	Timestamp    time.Time `json:"timestamp" db:"timestamp"`
	Method       string    `json:"method" db:"method"`
	Endpoint     string    `json:"endpoint" db:"endpoint"`
	Model        string    `json:"model" db:"model"`
	Backend      string    `json:"backend" db:"backend"`
	Status       int       `json:"status" db:"status"`
	Duration     int64     `json:"duration" db:"duration"` // milliseconds
	PromptTokens int       `json:"prompt_tokens" db:"prompt_tokens"`
	TotalTokens  int       `json:"total_tokens" db:"total_tokens"`
	RequestSize  int64     `json:"request_size" db:"request_size"`
	ResponseSize int64     `json:"response_size" db:"response_size"`
	UserAgent    string    `json:"user_agent" db:"user_agent"`
	ClientIP     string    `json:"client_ip" db:"client_ip"`
	RequestID    string    `json:"request_id" db:"request_id"`
	Streaming    bool      `json:"streaming" db:"streaming"`
	Error        string    `json:"error,omitempty" db:"error"`
}

// MetricsSummary represents aggregated metrics
type MetricsSummary struct {
	TotalRequests         int64          `json:"total_requests"`
	TotalTokens           int64          `json:"total_tokens"`
	TotalPromptTokens     int64          `json:"total_prompt_tokens"`
	TotalCompletionTokens int64          `json:"total_completion_tokens"`
	AvgResponseTime       float64        `json:"avg_response_time"`
	SuccessRate           float64        `json:"success_rate"`
	TopModels             []ModelUsage   `json:"top_models"`
	TopBackends           []BackendUsage `json:"top_backends"`
	HourlyStats           []HourlyStats  `json:"hourly_stats"`
	DailyStats            []DailyStats   `json:"daily_stats"`
}

// ModelUsage represents usage statistics for a specific model
type ModelUsage struct {
	Model        string  `json:"model"`
	Backend      string  `json:"backend"`
	RequestCount int64   `json:"request_count"`
	TokenCount   int64   `json:"token_count"`
	AvgTokens    float64 `json:"avg_tokens"`
}

// BackendUsage represents usage statistics for a specific backend
type BackendUsage struct {
	Backend      string  `json:"backend"`
	RequestCount int64   `json:"request_count"`
	TokenCount   int64   `json:"token_count"`
	AvgTokens    float64 `json:"avg_tokens"`
	SuccessRate  float64 `json:"success_rate"`
}

// HourlyStats represents metrics aggregated by hour
type HourlyStats struct {
	Hour         time.Time `json:"hour"`
	RequestCount int64     `json:"request_count"`
	TokenCount   int64     `json:"token_count"`
	ErrorCount   int64     `json:"error_count"`
}

// DailyStats represents metrics aggregated by day
type DailyStats struct {
	Date         time.Time `json:"date"`
	RequestCount int64     `json:"request_count"`
	TokenCount   int64     `json:"token_count"`
	ErrorCount   int64     `json:"error_count"`
}

// MetricsFilter represents filters for querying metrics
type MetricsFilter struct {
	StartTime *time.Time `json:"start_time,omitempty"`
	EndTime   *time.Time `json:"end_time,omitempty"`
	Model     string     `json:"model,omitempty"`
	Backend   string     `json:"backend,omitempty"`
	Method    string     `json:"method,omitempty"`
	Status    *int       `json:"status,omitempty"`
	MinTokens *int       `json:"min_tokens,omitempty"`
	MaxTokens *int       `json:"max_tokens,omitempty"`
	Streaming *bool      `json:"streaming,omitempty"`
	Limit     int        `json:"limit,omitempty"`
	Offset    int        `json:"offset,omitempty"`
	OrderBy   string     `json:"order_by,omitempty"`  // "timestamp", "duration", "tokens"
	OrderDir  string     `json:"order_dir,omitempty"` // "asc", "desc"
}

// MetricsConfig represents configuration for metrics collection
type MetricsConfig struct {
	Enabled             bool          `json:"enabled"`
	DatabasePath        string        `json:"database_path"`
	RetentionPeriod     time.Duration `json:"retention_period"`
	AggregationInterval time.Duration `json:"aggregation_interval"`
	MaxConnections      int           `json:"max_connections"`
	CollectRequestSize  bool          `json:"collect_request_size"`
	CollectResponseSize bool          `json:"collect_response_size"`
	CollectUserAgent    bool          `json:"collect_user_agent"`
	CollectClientIP     bool          `json:"collect_client_ip"`
	AnonymizeIPs        bool          `json:"anonymize_ips"`
}

// DefaultMetricsConfig returns default metrics configuration
func DefaultMetricsConfig() MetricsConfig {
	return MetricsConfig{
		Enabled:             true,
		DatabasePath:        "metrics.db",
		RetentionPeriod:     30 * 24 * time.Hour, // 30 days
		AggregationInterval: 1 * time.Hour,
		MaxConnections:      10,
		CollectRequestSize:  true,
		CollectResponseSize: true,
		CollectUserAgent:    true,
		CollectClientIP:     true,
		AnonymizeIPs:        true,
	}
}

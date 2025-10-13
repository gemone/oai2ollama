package services

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/gemone/oai2ollama/internal/models"
	"github.com/gofiber/fiber/v2/log"
	_ "github.com/mattn/go-sqlite3"
)

type MetricsService struct {
	db     *sql.DB
	config models.MetricsConfig
}

func NewMetricsService(config models.MetricsConfig) (*MetricsService, error) {
	if !config.Enabled {
		return &MetricsService{config: config}, nil
	}

	db, err := sql.Open("sqlite3", config.DatabasePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open metrics database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(config.MaxConnections)
	db.SetMaxIdleConns(config.MaxConnections / 2)

	service := &MetricsService{
		db:     db,
		config: config,
	}

	// Initialize database schema
	if err := service.initSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize metrics schema: %w", err)
	}

	// Start background cleanup and aggregation
	go service.startBackgroundTasks()

	return service, nil
}

func (ms *MetricsService) initSchema() error {
	createTableSQL := `
	CREATE TABLE IF NOT EXISTS api_metrics (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp DATETIME NOT NULL,
		method TEXT NOT NULL,
		endpoint TEXT NOT NULL,
		model TEXT,
		backend TEXT,
		status INTEGER NOT NULL,
		duration INTEGER NOT NULL,
		prompt_tokens INTEGER DEFAULT 0,
		total_tokens INTEGER DEFAULT 0,
		request_size INTEGER DEFAULT 0,
		response_size INTEGER DEFAULT 0,
		user_agent TEXT,
		client_ip TEXT,
		request_id TEXT,
		streaming BOOLEAN DEFAULT FALSE,
		error TEXT
	);

	CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON api_metrics(timestamp);
	CREATE INDEX IF NOT EXISTS idx_metrics_model ON api_metrics(model);
	CREATE INDEX IF NOT EXISTS idx_metrics_backend ON api_metrics(backend);
	CREATE INDEX IF NOT EXISTS idx_metrics_status ON api_metrics(status);
	CREATE INDEX IF NOT EXISTS idx_metrics_endpoint ON api_metrics(endpoint);
	`

	_, err := ms.db.Exec(createTableSQL)
	return err
}

func (ms *MetricsService) RecordMetric(metric *models.APIMetrics) error {
	if !ms.config.Enabled || ms.db == nil {
		return nil
	}

	// Anonymize IP if configured
	if ms.config.AnonymizeIPs && metric.ClientIP != "" {
		metric.ClientIP = ms.anonymizeIP(metric.ClientIP)
	}

	query := `
	INSERT INTO api_metrics (
		timestamp, method, endpoint, model, backend, status, duration,
		prompt_tokens, total_tokens, request_size, response_size,
		user_agent, client_ip, request_id, streaming, error
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	_, err := ms.db.Exec(query,
		metric.Timestamp, metric.Method, metric.Endpoint, metric.Model, metric.Backend,
		metric.Status, metric.Duration, metric.PromptTokens, metric.TotalTokens,
		metric.RequestSize, metric.ResponseSize, metric.UserAgent, metric.ClientIP,
		metric.RequestID, metric.Streaming, metric.Error,
	)

	if err != nil {
		log.Errorf("Failed to record metric: %v", err)
		return err
	}

	return nil
}

func (ms *MetricsService) GetMetrics(filter models.MetricsFilter) ([]models.APIMetrics, error) {
	if !ms.config.Enabled || ms.db == nil {
		return nil, nil
	}

	query := "SELECT * FROM api_metrics WHERE 1=1"
	args := []interface{}{}

	// Build WHERE clause
	if filter.StartTime != nil {
		query += " AND timestamp >= ?"
		args = append(args, filter.StartTime)
	}
	if filter.EndTime != nil {
		query += " AND timestamp <= ?"
		args = append(args, filter.EndTime)
	}
	if filter.Model != "" {
		query += " AND model = ?"
		args = append(args, filter.Model)
	}
	if filter.Backend != "" {
		query += " AND backend = ?"
		args = append(args, filter.Backend)
	}
	if filter.Method != "" {
		query += " AND method = ?"
		args = append(args, filter.Method)
	}
	if filter.Status != nil {
		query += " AND status = ?"
		args = append(args, *filter.Status)
	}
	if filter.MinTokens != nil {
		query += " AND total_tokens >= ?"
		args = append(args, *filter.MinTokens)
	}
	if filter.MaxTokens != nil {
		query += " AND total_tokens <= ?"
		args = append(args, *filter.MaxTokens)
	}
	if filter.Streaming != nil {
		query += " AND streaming = ?"
		args = append(args, *filter.Streaming)
	}

	// Add ORDER BY
	orderBy := "timestamp"
	if filter.OrderBy != "" {
		orderBy = filter.OrderBy
	}
	orderDir := "DESC"
	if filter.OrderDir != "" {
		orderDir = strings.ToUpper(filter.OrderDir)
	}
	query += fmt.Sprintf(" ORDER BY %s %s", orderBy, orderDir)

	// Add LIMIT and OFFSET
	if filter.Limit > 0 {
		query += " LIMIT ?"
		args = append(args, filter.Limit)
	}
	if filter.Offset > 0 {
		query += " OFFSET ?"
		args = append(args, filter.Offset)
	}

	rows, err := ms.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var metrics []models.APIMetrics
	for rows.Next() {
		var metric models.APIMetrics
		err := rows.Scan(
			&metric.ID, &metric.Timestamp, &metric.Method, &metric.Endpoint,
			&metric.Model, &metric.Backend, &metric.Status, &metric.Duration,
			&metric.PromptTokens, &metric.TotalTokens, &metric.RequestSize,
			&metric.ResponseSize, &metric.UserAgent, &metric.ClientIP,
			&metric.RequestID, &metric.Streaming, &metric.Error,
		)
		if err != nil {
			return nil, err
		}
		metrics = append(metrics, metric)
	}

	return metrics, nil
}

func (ms *MetricsService) GetSummary(filter models.MetricsFilter) (*models.MetricsSummary, error) {
	if !ms.config.Enabled || ms.db == nil {
		return &models.MetricsSummary{}, nil
	}

	summary := &models.MetricsSummary{}

	// Build base WHERE clause
	whereClause, args := ms.buildWhereClause(filter)

	// Get total requests and tokens
	query := fmt.Sprintf(`
		SELECT 
			COUNT(*) as total_requests,
			COALESCE(SUM(total_tokens), 0) as total_tokens,
			COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
			COALESCE(SUM(total_tokens - prompt_tokens), 0) as total_completion_tokens,
			COALESCE(AVG(duration), 0) as avg_response_time,
			COUNT(CASE WHEN status < 400 THEN 1 END) * 100.0 / COUNT(*) as success_rate
		FROM api_metrics WHERE %s
	`, whereClause)

	var avgResponseTime float64
	err := ms.db.QueryRow(query, args...).Scan(
		&summary.TotalRequests, &summary.TotalTokens, &summary.TotalPromptTokens,
		&summary.TotalCompletionTokens, &avgResponseTime, &summary.SuccessRate,
	)
	if err != nil {
		return nil, err
	}
	summary.AvgResponseTime = avgResponseTime

	// Get top models
	summary.TopModels, err = ms.getTopModels(filter, 10)
	if err != nil {
		log.Warnf("Failed to get top models: %v", err)
	}

	// Get top backends
	summary.TopBackends, err = ms.getTopBackends(filter, 10)
	if err != nil {
		log.Warnf("Failed to get top backends: %v", err)
	}

	// Get hourly stats (last 24 hours)
	summary.HourlyStats, err = ms.getHourlyStats(filter, 24)
	if err != nil {
		log.Warnf("Failed to get hourly stats: %v", err)
	}

	// Get daily stats (last 30 days)
	summary.DailyStats, err = ms.getDailyStats(filter, 30)
	if err != nil {
		log.Warnf("Failed to get daily stats: %v", err)
	}

	return summary, nil
}

func (ms *MetricsService) getTopModels(filter models.MetricsFilter, limit int) ([]models.ModelUsage, error) {
	whereClause, args := ms.buildWhereClause(filter)

	query := fmt.Sprintf(`
		SELECT 
			model,
			backend,
			COUNT(*) as request_count,
			COALESCE(SUM(total_tokens), 0) as token_count,
			COALESCE(AVG(total_tokens), 0) as avg_tokens
		FROM api_metrics 
		WHERE %s AND model IS NOT NULL AND model != ''
		GROUP BY model, backend
		ORDER BY request_count DESC
		LIMIT ?
	`, whereClause)

	args = append(args, limit)
	rows, err := ms.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var modelUsage []models.ModelUsage
	for rows.Next() {
		var mu models.ModelUsage
		err := rows.Scan(&mu.Model, &mu.Backend, &mu.RequestCount, &mu.TokenCount, &mu.AvgTokens)
		if err != nil {
			return nil, err
		}
		modelUsage = append(modelUsage, mu)
	}

	return modelUsage, nil
}

func (ms *MetricsService) getTopBackends(filter models.MetricsFilter, limit int) ([]models.BackendUsage, error) {
	whereClause, args := ms.buildWhereClause(filter)

	query := fmt.Sprintf(`
		SELECT 
			backend,
			COUNT(*) as request_count,
			COALESCE(SUM(total_tokens), 0) as token_count,
			COALESCE(AVG(total_tokens), 0) as avg_tokens,
			COUNT(CASE WHEN status < 400 THEN 1 END) * 100.0 / COUNT(*) as success_rate
		FROM api_metrics 
		WHERE %s AND backend IS NOT NULL AND backend != ''
		GROUP BY backend
		ORDER BY request_count DESC
		LIMIT ?
	`, whereClause)

	args = append(args, limit)
	rows, err := ms.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var backendUsage []models.BackendUsage
	for rows.Next() {
		var bu models.BackendUsage
		err := rows.Scan(&bu.Backend, &bu.RequestCount, &bu.TokenCount, &bu.AvgTokens, &bu.SuccessRate)
		if err != nil {
			return nil, err
		}
		backendUsage = append(backendUsage, bu)
	}

	return backendUsage, nil
}

func (ms *MetricsService) getHourlyStats(filter models.MetricsFilter, hours int) ([]models.HourlyStats, error) {
	whereClause, args := ms.buildWhereClause(filter)

	query := fmt.Sprintf(`
		SELECT 
			datetime(timestamp, 'start of hour') as hour,
			COUNT(*) as request_count,
			COALESCE(SUM(total_tokens), 0) as token_count,
			COUNT(CASE WHEN status >= 400 THEN 1 END) as error_count
		FROM api_metrics 
		WHERE %s AND timestamp >= datetime('now', '-%d hours')
		GROUP BY hour
		ORDER BY hour DESC
	`, whereClause, hours)

	rows, err := ms.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var stats []models.HourlyStats
	for rows.Next() {
		var stat models.HourlyStats
		err := rows.Scan(&stat.Hour, &stat.RequestCount, &stat.TokenCount, &stat.ErrorCount)
		if err != nil {
			return nil, err
		}
		stats = append(stats, stat)
	}

	return stats, nil
}

func (ms *MetricsService) getDailyStats(filter models.MetricsFilter, days int) ([]models.DailyStats, error) {
	whereClause, args := ms.buildWhereClause(filter)

	query := fmt.Sprintf(`
		SELECT 
			date(timestamp) as date,
			COUNT(*) as request_count,
			COALESCE(SUM(total_tokens), 0) as token_count,
			COUNT(CASE WHEN status >= 400 THEN 1 END) as error_count
		FROM api_metrics 
		WHERE %s AND timestamp >= date('now', '-%d days')
		GROUP BY date
		ORDER BY date DESC
	`, whereClause, days)

	args = append(args, days)
	rows, err := ms.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var stats []models.DailyStats
	for rows.Next() {
		var stat models.DailyStats
		err := rows.Scan(&stat.Date, &stat.RequestCount, &stat.TokenCount, &stat.ErrorCount)
		if err != nil {
			return nil, err
		}
		stats = append(stats, stat)
	}

	return stats, nil
}

func (ms *MetricsService) buildWhereClause(filter models.MetricsFilter) (string, []interface{}) {
	conditions := []string{"1=1"}
	args := []interface{}{}

	if filter.StartTime != nil {
		conditions = append(conditions, "timestamp >= ?")
		args = append(args, filter.StartTime)
	}
	if filter.EndTime != nil {
		conditions = append(conditions, "timestamp <= ?")
		args = append(args, filter.EndTime)
	}
	if filter.Model != "" {
		conditions = append(conditions, "model = ?")
		args = append(args, filter.Model)
	}
	if filter.Backend != "" {
		conditions = append(conditions, "backend = ?")
		args = append(args, filter.Backend)
	}
	if filter.Method != "" {
		conditions = append(conditions, "method = ?")
		args = append(args, filter.Method)
	}
	if filter.Status != nil {
		conditions = append(conditions, "status = ?")
		args = append(args, *filter.Status)
	}
	if filter.MinTokens != nil {
		conditions = append(conditions, "total_tokens >= ?")
		args = append(args, *filter.MinTokens)
	}
	if filter.MaxTokens != nil {
		conditions = append(conditions, "total_tokens <= ?")
		args = append(args, *filter.MaxTokens)
	}
	if filter.Streaming != nil {
		conditions = append(conditions, "streaming = ?")
		args = append(args, *filter.Streaming)
	}

	return strings.Join(conditions, " AND "), args
}

func (ms *MetricsService) anonymizeIP(ip string) string {
	// Simple IP anonymization - remove the last octet for IPv4 or last 64 bits for IPv6
	if strings.Contains(ip, ".") {
		// IPv4
		parts := strings.Split(ip, ".")
		if len(parts) == 4 {
			return fmt.Sprintf("%s.%s.%s.0", parts[0], parts[1], parts[2])
		}
	} else if strings.Contains(ip, ":") {
		// IPv6 - simple truncation
		parts := strings.Split(ip, ":")
		if len(parts) >= 4 {
			return strings.Join(parts[:4], ":") + "::"
		}
	}
	return "unknown"
}

func (ms *MetricsService) startBackgroundTasks() {
	ticker := time.NewTicker(ms.config.AggregationInterval)
	defer ticker.Stop()

	for range ticker.C {
		// Clean up old metrics
		if err := ms.cleanupOldMetrics(); err != nil {
			log.Errorf("Failed to cleanup old metrics: %v", err)
		}
	}
}

func (ms *MetricsService) cleanupOldMetrics() error {
	if ms.config.RetentionPeriod <= 0 {
		return nil
	}

	cutoffTime := time.Now().Add(-ms.config.RetentionPeriod)
	query := "DELETE FROM api_metrics WHERE timestamp < ?"

	result, err := ms.db.Exec(query, cutoffTime)
	if err != nil {
		return err
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected > 0 {
		log.Infof("Cleaned up %d old metric records", rowsAffected)
	}

	return nil
}

func (ms *MetricsService) Close() error {
	if ms.db != nil {
		return ms.db.Close()
	}
	return nil
}

// GetMetricsByModel returns metrics grouped by model
func (ms *MetricsService) GetMetricsByModel(model string, filter models.MetricsFilter) (*models.MetricsSummary, error) {
	filter.Model = model
	return ms.GetSummary(filter)
}

// GetMetricsByBackend returns metrics grouped by backend
func (ms *MetricsService) GetMetricsByBackend(backend string, filter models.MetricsFilter) (*models.MetricsSummary, error) {
	filter.Backend = backend
	return ms.GetSummary(filter)
}

// GetTokenUsage returns token usage statistics
func (ms *MetricsService) GetTokenUsage(filter models.MetricsFilter) (map[string]int64, error) {
	if !ms.config.Enabled || ms.db == nil {
		return make(map[string]int64), nil
	}

	whereClause, args := ms.buildWhereClause(filter)

	query := fmt.Sprintf(`
		SELECT 
			COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
			COALESCE(SUM(total_tokens - prompt_tokens), 0) as completion_tokens,
			COALESCE(SUM(total_tokens), 0) as total_tokens
		FROM api_metrics WHERE %s
	`, whereClause)

	var promptTokens, completionTokens, totalTokens int64
	err := ms.db.QueryRow(query, args...).Scan(&promptTokens, &completionTokens, &totalTokens)
	if err != nil {
		return nil, err
	}

	return map[string]int64{
		"prompt_tokens":     promptTokens,
		"completion_tokens": completionTokens,
		"total_tokens":      totalTokens,
	}, nil
}

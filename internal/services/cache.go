package services

import (
	"sync"
	"time"

	"github.com/gemone/oai2ollama/internal/models"
)

// ModelParseCache 模型解析结果缓存
type ModelParseCache struct {
	cache map[string]*ModelParseEntry
	mu    sync.RWMutex
}

// ModelParseEntry 模型解析缓存条目
type ModelParseEntry struct {
	Result    *models.ModelParseResult
	Timestamp time.Time
	TTL       time.Duration
}

// NewModelParseCache 创建新的模型解析缓存
func NewModelParseCache() *ModelParseCache {
	cache := &ModelParseCache{
		cache: make(map[string]*ModelParseEntry),
	}

	// 启动清理协程
	go cache.startCleanup()

	return cache
}

// Get 从缓存获取解析结果
func (c *ModelParseCache) Get(key string) (*models.ModelParseResult, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.cache[key]
	if !exists {
		return nil, false
	}

	// 检查是否过期
	if time.Since(entry.Timestamp) > entry.TTL {
		delete(c.cache, key)
		return nil, false
	}

	return entry.Result, true
}

// Set 设置缓存条目
func (c *ModelParseCache) Set(key string, result *models.ModelParseResult, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache[key] = &ModelParseEntry{
		Result:    result,
		Timestamp: time.Now(),
		TTL:       ttl,
	}
}

// Clear 清空缓存
func (c *ModelParseCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache = make(map[string]*ModelParseEntry)
}

// Size 返回缓存大小
func (c *ModelParseCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.cache)
}

// startCleanup 启动定期清理过期条目
func (c *ModelParseCache) startCleanup() {
	ticker := time.NewTicker(5 * time.Minute) // 每5分钟清理一次
	defer ticker.Stop()

	for range ticker.C {
		c.cleanup()
	}
}

// cleanup 清理过期条目
func (c *ModelParseCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, entry := range c.cache {
		if now.Sub(entry.Timestamp) > entry.TTL {
			delete(c.cache, key)
		}
	}
}

// ModelCapabilityCache 模型能力缓存
type ModelCapabilityCache struct {
	cache map[string]*CapabilityEntry
	mu    sync.RWMutex
}

// CapabilityEntry 能力缓存条目
type CapabilityEntry struct {
	Capabilities []string
	Timestamp    time.Time
	TTL          time.Duration
}

// NewModelCapabilityCache 创建新的模型能力缓存
func NewModelCapabilityCache() *ModelCapabilityCache {
	cache := &ModelCapabilityCache{
		cache: make(map[string]*CapabilityEntry),
	}

	// 启动清理协程
	go cache.startCleanup()

	return cache
}

// Get 从缓存获取能力信息
func (c *ModelCapabilityCache) Get(key string) ([]string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.cache[key]
	if !exists {
		return nil, false
	}

	// 检查是否过期
	if time.Since(entry.Timestamp) > entry.TTL {
		delete(c.cache, key)
		return nil, false
	}

	return entry.Capabilities, true
}

// Set 设置缓存条目
func (c *ModelCapabilityCache) Set(key string, capabilities []string, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache[key] = &CapabilityEntry{
		Capabilities: capabilities,
		Timestamp:    time.Now(),
		TTL:          ttl,
	}
}

// startCleanup 启动定期清理过期条目
func (c *ModelCapabilityCache) startCleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.cleanup()
	}
}

// cleanup 清理过期条目
func (c *ModelCapabilityCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, entry := range c.cache {
		if now.Sub(entry.Timestamp) > entry.TTL {
			delete(c.cache, key)
		}
	}
}

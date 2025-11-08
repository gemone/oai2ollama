package services

import (
	"fmt"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2/log"

	"github.com/gemone/oai2ollama/internal/config"
	"github.com/gemone/oai2ollama/internal/models"
)

type ModelCache struct {
	config    *config.Config
	clients   map[string]models.BackendClient
	converter *ModelConverter
	// 双缓存机制：主缓存和备用缓存
	cachedModels map[string]models.OllamaModelInfo // model name -> model info
	modelMap     map[string]string                 // model name -> backend name
	// 备用缓存，用于异步更新
	backupModels map[string]models.OllamaModelInfo
	backupMap    map[string]string
	mutex        sync.RWMutex
	lastUpdate   time.Time
	lastSuccess  time.Time // 最后成功更新时间
	ticker       *time.Ticker
	stopChan     chan bool
	refreshing   bool            // 是否正在刷新
	healthStatus map[string]bool // 后端健康状态
}

func NewModelCache(cfg *config.Config, clients map[string]models.BackendClient, converter *ModelConverter) *ModelCache {
	cache := &ModelCache{
		config:       cfg,
		clients:      clients,
		converter:    converter,
		cachedModels: make(map[string]models.OllamaModelInfo),
		modelMap:     make(map[string]string),
		backupModels: make(map[string]models.OllamaModelInfo),
		backupMap:    make(map[string]string),
		lastUpdate:   time.Time{},
		lastSuccess:  time.Time{},
		stopChan:     make(chan bool),
		healthStatus: make(map[string]bool),
	}

	// 初始化后端健康状态
	for backendName := range clients {
		cache.healthStatus[backendName] = true
	}

	// Start background refresh
	cache.startBackgroundRefresh()

	return cache
}

func (mc *ModelCache) startBackgroundRefresh() {
	// 初始同步刷新（首次启动时需要）
	mc.refreshModels()

	// Start ticker for periodic refresh (every 5 minutes)
	refreshInterval := 5 * time.Minute
	mc.ticker = time.NewTicker(refreshInterval)

	go func() {
		for {
			select {
			case <-mc.ticker.C:
				// 异步刷新，不阻塞请求
				go mc.asyncRefreshModels()
			case <-mc.stopChan:
				mc.ticker.Stop()
				return
			}
		}
	}()
}

func (mc *ModelCache) refreshModels() {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	log.Info("Refreshing model cache...")

	// Clear existing cache
	mc.cachedModels = make(map[string]models.OllamaModelInfo)
	mc.modelMap = make(map[string]string)

	// Track if we have any wildcard configuration
	hasWildcardBackend := make(map[string]bool)

	// Collect models from all enabled backends
	for backendName, backendClient := range mc.clients {
		openaiModels, err := backendClient.GetModels()
		if err != nil {
			log.Error("Failed to get models from backend %s: %v", backendName, err)

			// Even if GetModels() fails, we should still provide fallback models for backends without wildcard config
			wildcardEnabled := false
			for _, model := range mc.config.Models {
				if model.Enabled && model.Name == "*" && model.Backend == backendName {
					wildcardEnabled = true
					hasWildcardBackend[backendName] = true
					break
				}
			}

			// If no wildcard config and GetModels failed, create basic fallback models from config
			if !wildcardEnabled {
				log.Infof("Creating fallback models for backend %s from configuration", backendName)
				fallbackModels := mc.createConfigBasedFallbackModels(backendName)
				for _, model := range fallbackModels {
					mc.cachedModels[model.Name] = model
					mc.modelMap[model.Name] = backendName
				}
			}
			continue
		}

		// Check if wildcard configuration is enabled for this backend
		wildcardEnabled := false
		var wildcardConfig *config.ModelConfig
		for _, model := range mc.config.Models {
			if model.Enabled && model.Name == "*" && model.Backend == backendName {
				wildcardEnabled = true
				wildcardConfig = &model
				hasWildcardBackend[backendName] = true
				break
			}
		}

		// Get backend configuration for prefix handling
		var backendConfig *config.BackendConfig
		for _, backend := range mc.config.Backends {
			if backend.Name == backendName {
				backendConfig = &backend
				break
			}
		}

		// Convert OpenAI models to Ollama format
		var ollamaModels []models.OllamaModelInfo
		if wildcardEnabled {
			// If wildcard is enabled, convert all models from this backend
			ollamaModels = mc.convertAllOpenAIModels(openaiModels, backendName, wildcardConfig, backendConfig)
		} else {
			// Otherwise, use the regular converter (which will only convert matching models)
			ollamaModels = mc.converter.ConvertOpenAIModelsList(openaiModels)
			// Always add fallback models when no wildcard configuration exists
			// This ensures we have models available even without explicit * configuration
			fallbackModels := mc.convertBasicOpenAIModels(openaiModels, backendName, backendConfig)
			// Merge regular models and fallback models, avoiding duplicates
			ollamaModels = mc.mergeModelLists(ollamaModels, fallbackModels)
		}

		// Add to cache
		for _, model := range ollamaModels {
			mc.cachedModels[model.Name] = model
			// Map both the prefixed name and original name to the backend
			mc.modelMap[model.Name] = backendName
			if originalName, exists := model.Details["original_name"].(string); exists && originalName != model.Name {
				mc.modelMap[originalName] = backendName
			}
		}
	}

	// Process manually configured models (aliases)
	for _, model := range mc.config.Models {
		if !model.Enabled {
			continue
		}

		if model.Name == "*" {
			// For wildcard configuration, we've already handled all backend models above
			continue
		}

		// Check if this custom model name already exists in cache
		if _, exists := mc.cachedModels[model.Name]; exists {
			// Model name already exists, skip to avoid duplicates
			log.Debugf("Custom model name %s already exists, skipping", model.Name)
			continue
		}

		// Find the corresponding backend client
		backendClient, backendExists := mc.clients[model.Backend]
		if !backendExists {
			log.Warnf("Backend %s not found for custom model %s, skipping", model.Backend, model.Name)
			continue
		}

		// Check if the original model exists in the backend
		originalModelName := model.OriginalName
		if originalModelName == "" {
			originalModelName = model.Name
		}

		// Get all models from the backend to check if original model exists
		openaiModels, err := backendClient.GetModels()
		if err != nil {
			log.Warnf("Failed to get models from backend %s for custom model %s: %v", model.Backend, model.Name, err)
			continue
		}

		// Look for the original model in the backend
		var foundModel *models.OpenAIModel
		for _, openaiModel := range openaiModels {
			if openaiModel.ID == originalModelName {
				foundModel = &openaiModel
				break
			}
		}

		if foundModel == nil {
			log.Warnf("Original model %s not found in backend %s for custom model %s, skipping",
				originalModelName, model.Backend, model.Name)
			continue
		}

		// Verify the model actually exists using the already-fetched models list
		if !mc.verifyModelExistsFromList(openaiModels, originalModelName) {
			log.Warnf("Model %s verification failed in backend %s for custom model %s, skipping",
				originalModelName, model.Backend, model.Name)
			continue
		}

		// Get backend configuration for prefix handling
		var backendConfig *config.BackendConfig
		for _, backend := range mc.config.Backends {
			if backend.Name == model.Backend {
				backendConfig = &backend
				break
			}
		}

		// Convert the found model to Ollama format using the custom name
		customModel := mc.convertSingleModel(*foundModel, model.Name, model.Backend, backendConfig)

		// Override with manual configuration details
		if len(model.Capabilities) > 0 {
			customModel.Details["capabilities"] = model.Capabilities
		}
		if model.OriginalName != "" {
			customModel.Details["original_name"] = model.OriginalName
		}

		// Add to cache
		log.Debugf("Adding custom model: %s -> %s (backend: %s)", model.Name, originalModelName, model.Backend)
		mc.cachedModels[model.Name] = customModel
		mc.modelMap[model.Name] = model.Backend
	}

	mc.lastUpdate = time.Now()
	mc.lastSuccess = time.Now()
	log.Info("Model cache refreshed. Total models: %d", len(mc.cachedModels))
}

// asyncRefreshModels 异步刷新模型缓存，不阻塞读取请求
func (mc *ModelCache) asyncRefreshModels() {
	// 防止并发刷新
	if mc.refreshing {
		log.Debug("Model cache refresh already in progress, skipping")
		return
	}

	mc.refreshing = true
	defer func() {
		mc.refreshing = false
	}()

	log.Info("Starting async model cache refresh...")

	// 在备用缓存中构建新数据
	newModels := make(map[string]models.OllamaModelInfo)
	newModelMap := make(map[string]string)

	// 收集模型时考虑后端健康状态
	for backendName, backendClient := range mc.clients {
		// 检查后端健康状态
		if !mc.isBackendHealthy(backendName) {
			log.Warnf("Backend %s is unhealthy, skipping refresh", backendName)
			// 从现有缓存复制该后端的模型
			mc.copyExistingBackendModels(backendName, newModels, newModelMap)
			continue
		}

		openaiModels, err := backendClient.GetModels()
		if err != nil {
			log.Error("Failed to get models from backend %s: %v", backendName, err)
			// 标记后端为不健康
			mc.markBackendUnhealthy(backendName)

			// Even if GetModels() fails, we should still provide fallback models for backends without wildcard config
			wildcardEnabled := false
			for _, model := range mc.config.Models {
				if model.Enabled && model.Name == "*" && model.Backend == backendName {
					wildcardEnabled = true
					break
				}
			}

			// If no wildcard config and GetModels failed, create basic fallback models from config
			if !wildcardEnabled {
				log.Infof("Creating fallback models for backend %s from configuration (async refresh)", backendName)
				fallbackModels := mc.createConfigBasedFallbackModels(backendName)
				for _, model := range fallbackModels {
					newModels[model.Name] = model
					newModelMap[model.Name] = backendName
				}
			} else {
				// From existing cache copy the backend's models (only for wildcard backends)
				mc.copyExistingBackendModels(backendName, newModels, newModelMap)
			}
			continue
		}

		// 标记后端为健康
		mc.markBackendHealthy(backendName)

		// 处理模型...（复用原有逻辑）
		mc.processBackendModels(backendName, openaiModels, newModels, newModelMap)
	}

	// 原子性替换缓存
	mc.mutex.Lock()
	mc.backupModels = mc.cachedModels
	mc.backupMap = mc.modelMap
	mc.cachedModels = newModels
	mc.modelMap = newModelMap
	mc.lastUpdate = time.Now()
	mc.lastSuccess = time.Now()
	mc.mutex.Unlock()

	log.Info("Async model cache refresh completed. Total models: %d", len(newModels))
}

// isBackendHealthy 检查后端是否健康
func (mc *ModelCache) isBackendHealthy(backendName string) bool {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	return mc.healthStatus[backendName]
}

// markBackendHealthy 标记后端为健康
func (mc *ModelCache) markBackendHealthy(backendName string) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	mc.healthStatus[backendName] = true
}

// markBackendUnhealthy 标记后端为不健康
func (mc *ModelCache) markBackendUnhealthy(backendName string) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	mc.healthStatus[backendName] = false
}

// copyExistingBackendModels 从现有缓存复制后端模型
func (mc *ModelCache) copyExistingBackendModels(backendName string, newModels map[string]models.OllamaModelInfo, newModelMap map[string]string) {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	for modelName, model := range mc.cachedModels {
		if backend, exists := mc.modelMap[modelName]; exists && backend == backendName {
			newModels[modelName] = model
			newModelMap[modelName] = backendName
		}
	}
}

// processBackendModels 处理单个后端的模型
func (mc *ModelCache) processBackendModels(backendName string, openaiModels []models.OpenAIModel, newModels map[string]models.OllamaModelInfo, newModelMap map[string]string) {
	// 检查通配符配置
	wildcardEnabled := false
	var wildcardConfig *config.ModelConfig
	for _, model := range mc.config.Models {
		if model.Enabled && model.Name == "*" && model.Backend == backendName {
			wildcardEnabled = true
			wildcardConfig = &model
			break
		}
	}

	var backendConfig *config.BackendConfig
	for _, backend := range mc.config.Backends {
		if backend.Name == backendName {
			backendConfig = &backend
			break
		}
	}

	var ollamaModels []models.OllamaModelInfo
	if wildcardEnabled {
		ollamaModels = mc.convertAllOpenAIModels(openaiModels, backendName, wildcardConfig, backendConfig)
	} else {
		ollamaModels = mc.converter.ConvertOpenAIModelsList(openaiModels)
		// Always add fallback models when no wildcard configuration exists
		// This ensures we have models available even without explicit * configuration
		fallbackModels := mc.convertBasicOpenAIModels(openaiModels, backendName, backendConfig)
		// Merge regular models and fallback models, avoiding duplicates
		ollamaModels = mc.mergeModelLists(ollamaModels, fallbackModels)
	}

	// 添加到新缓存
	for _, model := range ollamaModels {
		newModels[model.Name] = model
		newModelMap[model.Name] = backendName
		if originalName, exists := model.Details["original_name"].(string); exists && originalName != model.Name {
			newModelMap[originalName] = backendName
		}
	}

	// 处理手动配置的模型
	mc.processManualModels(backendName, newModels, newModelMap)
}

// processManualModels 处理手动配置的模型
func (mc *ModelCache) processManualModels(backendName string, newModels map[string]models.OllamaModelInfo, newModelMap map[string]string) {
	for _, model := range mc.config.Models {
		if !model.Enabled || model.Backend != backendName {
			continue
		}

		if model.Name == "*" {
			continue // 通配符已处理
		}

		// 检查是否已存在
		if _, exists := newModels[model.Name]; exists {
			continue
		}

		// 获取后端客户端并验证模型
		backendClient, exists := mc.clients[backendName]
		if !exists {
			continue
		}

		openaiModels, err := backendClient.GetModels()
		if err != nil {
			continue
		}

		var foundModel *models.OpenAIModel
		originalModelName := model.OriginalName
		if originalModelName == "" {
			originalModelName = model.Name
		}

		for _, openaiModel := range openaiModels {
			if openaiModel.ID == originalModelName {
				foundModel = &openaiModel
				break
			}
		}

		if foundModel == nil {
			continue
		}

		var backendConfig *config.BackendConfig
		for _, backend := range mc.config.Backends {
			if backend.Name == backendName {
				backendConfig = &backend
				break
			}
		}

		customModel := mc.convertSingleModel(*foundModel, model.Name, backendName, backendConfig)

		if len(model.Capabilities) > 0 {
			customModel.Details["capabilities"] = model.Capabilities
		}
		if model.OriginalName != "" {
			customModel.Details["original_name"] = model.OriginalName
		}

		newModels[model.Name] = customModel
		newModelMap[model.Name] = backendName
	}
}

func (mc *ModelCache) GetAllModels() []models.OllamaModelInfo {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	var models []models.OllamaModelInfo
	for _, model := range mc.cachedModels {
		models = append(models, model)
	}

	return models
}

func (mc *ModelCache) GetModel(modelName string) (models.OllamaModelInfo, bool) {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	model, exists := mc.cachedModels[modelName]
	return model, exists
}

func (mc *ModelCache) GetBackendForModel(modelName string) (string, error) {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	// First try exact match
	backend, exists := mc.modelMap[modelName]
	if exists {
		return backend, nil
	}

	// If not found, try to find a suitable backend with fallback model
	log.Debugf("Model '%s' not found in cache, trying fallback", modelName)

	// Try to match with any backend that has fallback models
	for backendName := range mc.clients {
		// Check if this backend has any models that could serve as fallback
		for cachedModelName, cachedBackend := range mc.modelMap {
			if cachedBackend != backendName {
				continue
			}

			// Check if this is a fallback model
			if model, exists := mc.cachedModels[cachedModelName]; exists {
				if isFallback, ok := model.Details["fallback_model"].(bool); ok && isFallback {
					// Extract original name from fallback model
					if originalName, exists := model.Details["original_name"].(string); exists {
						// Check if the requested model matches the original name (with or without prefix)
						if modelName == originalName || modelName == cachedModelName {
							log.Debugf("Found fallback backend '%s' for model '%s' (cached as '%s')", backendName, modelName, cachedModelName)
							// Add mapping for future requests
							mc.modelMap[modelName] = backendName
							return backendName, nil
						}
					}
				}
			}
		}
	}

	// If still not found, try to use any available backend as last resort
	log.Debugf("No specific fallback found for model '%s', trying any available backend", modelName)
	for backendName := range mc.clients {
		log.Debugf("Using backend '%s' as fallback for model '%s'", backendName, modelName)
		// Note: We can't add mapping here because we only have read lock
		// The mapping will be added when the model is actually requested and processed
		return backendName, nil
	}

	return "", fmt.Errorf("model not found: %s", modelName)
}

// AddModelMapping 线程安全地添加模型映射
func (mc *ModelCache) AddModelMapping(modelName, backendName string) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	mc.modelMap[modelName] = backendName
}

func (mc *ModelCache) ForceRefresh() {
	mc.refreshModels()
}

func (mc *ModelCache) Stop() {
	close(mc.stopChan)
}

func (mc *ModelCache) convertAllOpenAIModels(openaiModels []models.OpenAIModel, backendName string, wildcardConfig *config.ModelConfig, backendConfig *config.BackendConfig) []models.OllamaModelInfo {
	var ollamaModels []models.OllamaModelInfo

	for _, openaiModel := range openaiModels {
		// Apply prefix if configured
		var modelName string
		var originalName string

		if backendConfig != nil && backendConfig.ModelPrefix != nil && backendConfig.ModelPrefix.Enabled {
			prefix := backendConfig.ModelPrefix.Prefix
			if prefix == "" {
				prefix = backendConfig.Name
			}
			separator := backendConfig.ModelPrefix.Separator
			if separator == "" {
				separator = "/"
			}
			modelName = fmt.Sprintf("%s%s%s", prefix, separator, openaiModel.ID)
			originalName = openaiModel.ID
		} else {
			modelName = openaiModel.ID
			originalName = openaiModel.ID
		}

		// Generate SHA256 digest
		digest := mc.converter.GenerateModelDigest(modelName, openaiModel.Created)

		// Estimate model size
		size := mc.converter.EstimateModelSize(openaiModel.ID)

		// Extract parameter size
		parameterSize := mc.converter.ExtractParameterSize(openaiModel.ID)

		ollamaModel := models.OllamaModelInfo{
			Name:       modelName,
			Model:      modelName,
			ModifiedAt: time.Unix(openaiModel.Created, 0),
			Size:       size,
			Digest:     digest,
			Details: map[string]interface{}{
				"parent_model":       "",
				"format":             "gguf",
				"family":             mc.converter.DetermineModelFamily(openaiModel.ID),
				"families":           []string{mc.converter.DetermineModelFamily(openaiModel.ID)},
				"parameter_size":     parameterSize,
				"quantization_level": "unknown",
				"backend":            backendName,
				"original_name":      originalName,
				"openai_object":      openaiModel.Object,
				"openai_created":     openaiModel.Created,
				"openai_owned_by":    openaiModel.OwnedBy,
			},
		}

		// Add capabilities from wildcard config
		if len(wildcardConfig.Capabilities) > 0 {
			ollamaModel.Details["capabilities"] = wildcardConfig.Capabilities
		} else {
			// Default capabilities
			capabilities := mc.converter.getModelCapabilities(openaiModel.ID, backendName)
			ollamaModel.Details["capabilities"] = capabilities
		}

		ollamaModels = append(ollamaModels, ollamaModel)
	}

	return ollamaModels
}

// mergeModelLists merges two model lists, avoiding duplicates
func (mc *ModelCache) mergeModelLists(primary, fallback []models.OllamaModelInfo) []models.OllamaModelInfo {
	seen := make(map[string]bool)
	var result []models.OllamaModelInfo

	// Add primary models first
	for _, model := range primary {
		if !seen[model.Name] {
			seen[model.Name] = true
			result = append(result, model)
		}
	}

	// Add fallback models that aren't already in primary
	for _, model := range fallback {
		if !seen[model.Name] {
			seen[model.Name] = true
			result = append(result, model)
		}
	}

	return result
}

// createConfigBasedFallbackModels creates basic fallback models from configuration when GetModels() fails
func (mc *ModelCache) createConfigBasedFallbackModels(backendName string) []models.OllamaModelInfo {
	var fallbackModels []models.OllamaModelInfo

	// Get backend configuration for prefix handling
	var backendConfig *config.BackendConfig
	for _, backend := range mc.config.Backends {
		if backend.Name == backendName {
			backendConfig = &backend
			break
		}
	}

	// Look for models configured for this backend in the config
	for _, model := range mc.config.Models {
		if !model.Enabled || model.Backend != backendName || model.Name == "*" {
			continue
		}

		// Create a basic OpenAI model structure
		originalName := model.OriginalName
		if originalName == "" {
			originalName = model.Name
		}

		// Apply prefix if configured
		var modelName string
		if backendConfig != nil && backendConfig.ModelPrefix != nil && backendConfig.ModelPrefix.Enabled {
			// For manually configured models, use the configured name directly
			// Don't apply prefix automatically since user has already specified the exact name they want
			modelName = model.Name
		} else {
			modelName = model.Name
		}

		// Create basic model info
		digest := mc.converter.GenerateModelDigest(modelName, time.Now().Unix())
		size := int64(8589934592) // Default 8GB size

		ollamaModel := models.OllamaModelInfo{
			Name:       modelName,
			Model:      modelName,
			ModifiedAt: time.Now(),
			Size:       size,
			Digest:     digest,
			Details: map[string]interface{}{
				"parent_model":       "",
				"format":             "gguf",
				"family":             extractFamily(originalName),
				"families":           []string{extractFamily(originalName)},
				"parameter_size":     "unknown",
				"quantization_level": "unknown",
				"backend":            backendName,
				"original_name":      originalName,
				"fallback_model":     true,
				"config_based":       true,
			},
		}

		// Add capabilities from config
		if len(model.Capabilities) > 0 {
			ollamaModel.Details["capabilities"] = model.Capabilities
		} else {
			// Default capabilities
			capabilities := []string{"completion", "chat"}
			ollamaModel.Details["capabilities"] = capabilities
		}

		fallbackModels = append(fallbackModels, ollamaModel)
		log.Debugf("Created config-based fallback model: %s -> %s (backend: %s)", modelName, originalName, backendName)
	}

	// If no models found in config for this backend, create a default fallback model
	if len(fallbackModels) == 0 {
		defaultModelName := fmt.Sprintf("%s/default-model", backendName)
		if backendConfig != nil && backendConfig.ModelPrefix != nil && backendConfig.ModelPrefix.Enabled {
			prefix := backendConfig.ModelPrefix.Prefix
			if prefix == "" {
				prefix = backendConfig.Name
			}
			separator := backendConfig.ModelPrefix.Separator
			if separator == "" {
				separator = "/"
			}
			defaultModelName = fmt.Sprintf("%s%sdefault-model", prefix, separator)
		}

		digest := mc.converter.GenerateModelDigest(defaultModelName, time.Now().Unix())

		defaultModel := models.OllamaModelInfo{
			Name:       defaultModelName,
			Model:      defaultModelName,
			ModifiedAt: time.Now(),
			Size:       int64(8589934592), // Default 8GB
			Digest:     digest,
			Details: map[string]interface{}{
				"parent_model":       "",
				"format":             "gguf",
				"family":             "unknown",
				"families":           []string{"unknown"},
				"parameter_size":     "unknown",
				"quantization_level": "unknown",
				"backend":            backendName,
				"original_name":      "default-model",
				"fallback_model":     true,
				"config_based":       true,
				"capabilities":       []string{"completion", "chat"},
			},
		}

		fallbackModels = append(fallbackModels, defaultModel)
		log.Infof("Created default fallback model for backend %s: %s", backendName, defaultModelName)
	}

	return fallbackModels
}

// convertBasicOpenAIModels converts all OpenAI models with basic info for fallback
func (mc *ModelCache) convertBasicOpenAIModels(openaiModels []models.OpenAIModel, backendName string, backendConfig *config.BackendConfig) []models.OllamaModelInfo {
	var ollamaModels []models.OllamaModelInfo

	for _, openaiModel := range openaiModels {
		// Apply prefix if configured
		var modelName string
		var originalName string

		if backendConfig != nil && backendConfig.ModelPrefix != nil && backendConfig.ModelPrefix.Enabled {
			prefix := backendConfig.ModelPrefix.Prefix
			if prefix == "" {
				prefix = backendConfig.Name
			}
			separator := backendConfig.ModelPrefix.Separator
			if separator == "" {
				separator = "/"
			}
			modelName = fmt.Sprintf("%s%s%s", prefix, separator, openaiModel.ID)
			originalName = openaiModel.ID
		} else {
			modelName = openaiModel.ID
			originalName = openaiModel.ID
		}

		// Generate SHA256 digest
		digest := mc.converter.GenerateModelDigest(modelName, openaiModel.Created)

		// Estimate model size
		size := mc.converter.EstimateModelSize(openaiModel.ID)

		// Extract parameter size
		parameterSize := mc.converter.ExtractParameterSize(openaiModel.ID)

		ollamaModel := models.OllamaModelInfo{
			Name:       modelName,
			Model:      modelName,
			ModifiedAt: time.Unix(openaiModel.Created, 0),
			Size:       size,
			Digest:     digest,
			Details: map[string]interface{}{
				"parent_model":       "",
				"format":             "gguf",
				"family":             mc.converter.DetermineModelFamily(openaiModel.ID),
				"families":           []string{mc.converter.DetermineModelFamily(openaiModel.ID)},
				"parameter_size":     parameterSize,
				"quantization_level": "unknown",
				"backend":            backendName,
				"original_name":      originalName,
				"openai_object":      openaiModel.Object,
				"openai_created":     openaiModel.Created,
				"openai_owned_by":    openaiModel.OwnedBy,
				"fallback_model":     true, // Mark this as a fallback model
			},
		}

		// Add basic capabilities for fallback models
		capabilities := []string{"completion", "chat", "embeddings"}
		ollamaModel.Details["capabilities"] = capabilities

		ollamaModels = append(ollamaModels, ollamaModel)
	}

	return ollamaModels
}

func (mc *ModelCache) GetCacheInfo() map[string]interface{} {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	// 计算健康后端数量
	healthyBackends := 0
	for _, healthy := range mc.healthStatus {
		if healthy {
			healthyBackends++
		}
	}

	return map[string]interface{}{
		"total_models":     len(mc.cachedModels),
		"last_update":      mc.lastUpdate,
		"last_success":     mc.lastSuccess,
		"backends":         len(mc.clients),
		"healthy_backends": healthyBackends,
		"refreshing":       mc.refreshing,
		"health_status":    mc.healthStatus,
	}
}

// convertSingleModel converts a single OpenAI model to Ollama format with custom name
func (mc *ModelCache) convertSingleModel(openaiModel models.OpenAIModel, customName, backendName string, backendConfig *config.BackendConfig) models.OllamaModelInfo {
	// For custom models, use the provided custom name instead of applying prefix logic
	modelName := customName
	originalName := openaiModel.ID

	// Create a SHA256 digest using the existing converter implementation
	digest := mc.converter.GenerateModelDigest(openaiModel.ID, openaiModel.Created)

	return models.OllamaModelInfo{
		Name:       modelName,
		Model:      modelName,
		Digest:     digest,
		ModifiedAt: time.Now(),
		Size:       mc.converter.EstimateModelSize(openaiModel.ID), // Use existing converter implementation
		Details: map[string]interface{}{
			"parent_model":       "",
			"format":             "gguf",
			"family":             extractFamily(openaiModel.ID),
			"families":           []string{extractFamily(openaiModel.ID)},
			"parameter_size":     mc.converter.ExtractParameterSize(openaiModel.ID), // Use existing implementation
			"quantization_level": "unknown",
			"backend":            backendName,
			"original_name":      originalName,
			"openai_created":     openaiModel.Created,
			"openai_object":      openaiModel.Object,
			"openai_owned_by":    openaiModel.OwnedBy,
		},
	}
}

// verifyModelExistsFromList verifies if a model exists in a given models slice
func (mc *ModelCache) verifyModelExistsFromList(models []models.OpenAIModel, modelName string) bool {
	for _, model := range models {
		if model.ID == modelName {
			log.Debugf("Model verification successful for %s", modelName)
			return true
		}
	}
	log.Debugf("Model verification failed for %s: model not found in current backend models", modelName)
	return false
}

const familyPrefixLength = 3

// extractFamily returns a string representing the "family" of a model based on its name.
// The function extracts the first three characters of the modelName as the family prefix,
// which is a simple heuristic and may not always correspond to a meaningful family.
// If the modelName is shorter than three characters, it returns "unknown".
// This logic can be enhanced to use more sophisticated family detection if needed.
func extractFamily(modelName string) string {
	// Simple extraction - could be enhanced
	if len(modelName) > familyPrefixLength {
		return modelName[:familyPrefixLength]
	}
	return "unknown"
}

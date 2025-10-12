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
	config       *config.Config
	clients      map[string]models.BackendClient
	converter    *ModelConverter
	cachedModels map[string]models.OllamaModelInfo // model name -> model info
	modelMap     map[string]string                 // model name -> backend name
	mutex        sync.RWMutex
	lastUpdate   time.Time
	ticker       *time.Ticker
	stopChan     chan bool
}

func NewModelCache(cfg *config.Config, clients map[string]models.BackendClient, converter *ModelConverter) *ModelCache {
	cache := &ModelCache{
		config:       cfg,
		clients:      clients,
		converter:    converter,
		cachedModels: make(map[string]models.OllamaModelInfo),
		modelMap:     make(map[string]string),
		lastUpdate:   time.Time{},
		stopChan:     make(chan bool),
	}

	// Start background refresh
	cache.startBackgroundRefresh()

	return cache
}

func (mc *ModelCache) startBackgroundRefresh() {
	// Initial refresh
	mc.refreshModels()

	// Start ticker for periodic refresh (every 5 minutes)
	refreshInterval := 5 * time.Minute
	mc.ticker = time.NewTicker(refreshInterval)

	go func() {
		for {
			select {
			case <-mc.ticker.C:
				mc.refreshModels()
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

	// Collect models from all enabled backends
	for backendName, backendClient := range mc.clients {
		openaiModels, err := backendClient.GetModels()
		if err != nil {
			log.Error("Failed to get models from backend %s: %v", backendName, err)
			continue
		}

		// Check if wildcard configuration is enabled for this backend
		wildcardEnabled := false
		var wildcardConfig *config.ModelConfig
		for _, model := range mc.config.Models {
			if model.Enabled && model.Name == "*" && model.Backend == backendName {
				wildcardEnabled = true
				wildcardConfig = &model
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
	log.Info("Model cache refreshed. Total models: %d", len(mc.cachedModels))
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

	backend, exists := mc.modelMap[modelName]
	if !exists {
		return "", fmt.Errorf("model not found: %s", modelName)
	}

	return backend, nil
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

func (mc *ModelCache) GetCacheInfo() map[string]interface{} {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	return map[string]interface{}{
		"total_models": len(mc.cachedModels),
		"last_update":  mc.lastUpdate,
		"backends":     len(mc.clients),
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

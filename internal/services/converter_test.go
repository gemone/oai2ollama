package services

import (
	"fmt"
	"testing"
	"time"

	"github.com/gemone/oai2ollama/internal/config"
	"github.com/gemone/oai2ollama/internal/models"
)

// BenchmarkModelConverter 测试 ModelConverter 的性能
func BenchmarkModelConverter(b *testing.B) {
	// 创建测试配置
	cfg := &config.Config{
		Backends: []config.BackendConfig{
			{
				Name:    "openai",
				Enabled: true,
				ModelPrefix: &config.ModelPrefixConfig{
					Enabled:   true,
					Prefix:    "openai",
					Separator: "/",
				},
			},
		},
		Models: []config.ModelConfig{
			{
				Name:         "openai/gpt-4",
				Backend:      "openai",
				OriginalName: "gpt-4",
				Enabled:      true,
				Capabilities: []string{"completion", "tools"},
			},
		},
	}

	converter := NewModelConverter(cfg)

	// 测试模型名称
	testModels := []string{
		"openai/gpt-4",
		"openai/gpt-4-vision",
		"openai/gpt-3.5-turbo",
		"openai/claude-3-opus",
		"openai/llama-2-70b",
		"openai/text-embedding-ada-002",
		"unknown-model",
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			model := testModels[i%len(testModels)]
			_, _ = converter.ParseModelName(model) // 忽略错误用于基准测试
			converter.getModelCapabilities(model, "openai")
			converter.EstimateModelSize(model)
			converter.ExtractParameterSize(model)
			converter.DetermineModelFamily(model)
			i++
		}
	})
}

// BenchmarkPatternMatcher 测试模式匹配器的性能
func BenchmarkPatternMatcher(b *testing.B) {
	matcher := GetGlobalPatternMatcher()

	testModels := []string{
		"gpt-4",
		"gpt-4-vision",
		"gpt-3.5-turbo",
		"claude-3-opus",
		"claude-3-sonnet",
		"llama-2-70b",
		"llama-2-13b",
		"text-embedding-ada-002",
		"unknown-model",
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			model := testModels[i%len(testModels)]
			matcher.MatchModel(model)
			i++
		}
	})
}

// BenchmarkCachePerformance 测试缓存性能
func BenchmarkCachePerformance(b *testing.B) {
	parseCache := NewModelParseCache()
	capabilityCache := NewModelCapabilityCache()

	// 预填充缓存
	testResult := &models.ModelParseResult{
		Backend:      "openai",
		OriginalName: "gpt-4",
		DisplayName:  "openai/gpt-4",
		ExactMatch:   true,
	}
	parseCache.Set("test-model", testResult, time.Hour)
	capabilityCache.Set("test-model:openai", []string{"completion", "tools"}, time.Hour)

	b.ResetTimer()
	b.Run("ParseCache_Get", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			parseCache.Get("test-model")
		}
	})

	b.Run("CapabilityCache_Get", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			capabilityCache.Get("test-model:openai")
		}
	})

	b.Run("ParseCache_Set", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("model-%d", i)
			parseCache.Set(key, testResult, time.Hour)
		}
	})
}

// TestOptimizationEffectiveness 测试优化效果
func TestOptimizationEffectiveness(t *testing.T) {
	// 创建测试配置
	cfg := &config.Config{
		Backends: []config.BackendConfig{
			{
				Name:    "openai",
				Enabled: true,
				ModelPrefix: &config.ModelPrefixConfig{
					Enabled:   true,
					Prefix:    "openai",
					Separator: "/",
				},
			},
		},
		Models: []config.ModelConfig{
			{
				Name:         "openai/gpt-4",
				Backend:      "openai",
				OriginalName: "gpt-4",
				Enabled:      true,
				Capabilities: []string{"completion", "tools"},
			},
		},
	}

	converter := NewModelConverter(cfg)
	testModel := "openai/gpt-4"

	// 测试多次调用的性能（应该受益于缓存）
	start := time.Now()
	for i := 0; i < 1000; i++ {
		_, _ = converter.ParseModelName(testModel) // 忽略错误用于性能测试
		converter.getModelCapabilities(testModel, "openai")
	}
	duration := time.Since(start)

	t.Logf("1000次调用耗时: %v (平均每次: %v)", duration, duration/1000)

	// 验证结果正确性
	result, err := converter.ParseModelName(testModel)
	if err != nil {
		t.Fatalf("ParseModelName 失败: %v", err)
	}

	if result.Backend != "openai" {
		t.Errorf("期望 backend 为 'openai', 实际为 '%s'", result.Backend)
	}

	if result.OriginalName != "gpt-4" {
		t.Errorf("期望 original_name 为 'gpt-4', 实际为 '%s'", result.OriginalName)
	}

	capabilities := converter.getModelCapabilities(testModel, "openai")
	expectedCapabilities := []string{"completion", "tools"}
	if len(capabilities) != len(expectedCapabilities) {
		t.Errorf("期望 capabilities 长度为 %d, 实际为 %d", len(expectedCapabilities), len(capabilities))
	}

	size := converter.EstimateModelSize(testModel)
	expectedSize := int64(38 * 1024 * 1024 * 1024) // 38GB
	if size != expectedSize {
		t.Errorf("期望模型大小为 %d, 实际为 %d", expectedSize, size)
	}

	family := converter.DetermineModelFamily(testModel)
	if family != "gpt4" {
		t.Errorf("期望模型家族为 'gpt4', 实际为 '%s'", family)
	}

	paramSize := converter.ExtractParameterSize(testModel)
	if paramSize != "unknown" {
		t.Errorf("期望参数大小为 'unknown', 实际为 '%s'", paramSize)
	}
}

// TestPatternMatcherAccuracy 测试模式匹配器的准确性
func TestPatternMatcherAccuracy(t *testing.T) {
	matcher := GetGlobalPatternMatcher()

	testCases := []struct {
		modelID        string
		expectedFamily string
		expectedCaps   []string
	}{
		{"gpt-4", "gpt4", []string{"completion", "tools"}},
		{"gpt-4-vision", "gpt4", []string{"completion", "tools", "vision"}},
		{"gpt-3.5-turbo", "gpt3.5", []string{"completion", "tools"}},
		{"claude-3-opus", "claude", []string{"completion", "tools", "vision"}},
		{"llama-2-70b", "llama", []string{"completion", "tools"}},
		{"text-embedding-ada-002", "embedding", []string{"embedding"}},
		{"o1-preview", "thinking", []string{"completion", "thinking"}},
		{"unknown-model", "unknown", []string{"completion"}},
	}

	for _, tc := range testCases {
		result := matcher.MatchModel(tc.modelID)

		if result.Family != tc.expectedFamily {
			t.Errorf("模型 %s: 期望家族 '%s', 实际 '%s'", tc.modelID, tc.expectedFamily, result.Family)
		}

		if len(result.Capabilities) != len(tc.expectedCaps) {
			t.Errorf("模型 %s: 期望能力数量 %d, 实际 %d", tc.modelID, len(tc.expectedCaps), len(result.Capabilities))
			continue
		}

		for i, cap := range tc.expectedCaps {
			if i >= len(result.Capabilities) || result.Capabilities[i] != cap {
				t.Errorf("模型 %s: 期望能力 '%s', 实际 '%s'", tc.modelID, cap, result.Capabilities[i])
			}
		}
	}
}

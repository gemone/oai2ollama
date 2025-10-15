package services

import (
	"strings"
	"sync"
)

// ModelPatternMatcher 模型模式匹配器，用于优化重复的字符串匹配逻辑
type ModelPatternMatcher struct {
	// 预编译的模型模式
	patterns map[string][]ModelPattern
	mu       sync.RWMutex
}

// ModelPattern 定义模型匹配模式
type ModelPattern struct {
	Keywords     []string
	Capabilities []string
	Family       string
	SizeHint     int64 // 字节为单位
	ParamSize    string
}

// NewModelPatternMatcher 创建新的模型模式匹配器
func NewModelPatternMatcher() *ModelPatternMatcher {
	return &ModelPatternMatcher{
		patterns: make(map[string][]ModelPattern),
	}
}

// InitializePatterns 初始化所有模型模式
func (mpm *ModelPatternMatcher) InitializePatterns() {
	mpm.mu.Lock()
	defer mpm.mu.Unlock()

	// GPT-4 模式
	mpm.patterns["gpt-4"] = []ModelPattern{
		{
			Keywords:     []string{"gpt-4"},
			Capabilities: []string{"completion", "tools"},
			Family:       "gpt4",
			SizeHint:     38 * 1024 * 1024 * 1024, // 38GB
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"gpt-4", "32k"},
			Capabilities: []string{"completion", "tools"},
			Family:       "gpt4",
			SizeHint:     78 * 1024 * 1024 * 1024, // 78GB
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"gpt-4-vision"},
			Capabilities: []string{"completion", "tools", "vision"},
			Family:       "gpt4",
			SizeHint:     38 * 1024 * 1024 * 1024,
			ParamSize:    "unknown",
		},
	}

	// GPT-3.5 模式
	mpm.patterns["gpt-3.5"] = []ModelPattern{
		{
			Keywords:     []string{"gpt-3.5"},
			Capabilities: []string{"completion", "tools"},
			Family:       "gpt3.5",
			SizeHint:     6 * 1024 * 1024 * 1024, // 6GB
			ParamSize:    "unknown",
		},
	}

	// Claude 模式
	mpm.patterns["claude"] = []ModelPattern{
		{
			Keywords:     []string{"claude"},
			Capabilities: []string{"completion", "tools"},
			Family:       "claude",
			SizeHint:     10 * 1024 * 1024 * 1024, // 默认10GB
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"claude-3"},
			Capabilities: []string{"completion", "tools", "vision"},
			Family:       "claude",
			SizeHint:     10 * 1024 * 1024 * 1024,
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"claude-3", "opus"},
			Capabilities: []string{"completion", "tools", "vision"},
			Family:       "claude",
			SizeHint:     30 * 1024 * 1024 * 1024, // 30GB
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"claude-3", "sonnet"},
			Capabilities: []string{"completion", "tools", "vision"},
			Family:       "claude",
			SizeHint:     15 * 1024 * 1024 * 1024, // 15GB
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"claude-3", "haiku"},
			Capabilities: []string{"completion", "tools", "vision"},
			Family:       "claude",
			SizeHint:     10 * 1024 * 1024 * 1024, // 10GB
			ParamSize:    "unknown",
		},
	}

	// Llama 模式
	mpm.patterns["llama"] = []ModelPattern{
		{
			Keywords:     []string{"llama"},
			Capabilities: []string{"completion", "tools"},
			Family:       "llama",
			SizeHint:     10 * 1024 * 1024 * 1024, // 默认10GB
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"llama", "70b", "65b"},
			Capabilities: []string{"completion", "tools"},
			Family:       "llama",
			SizeHint:     140 * 1024 * 1024 * 1024, // 140GB
			ParamSize:    "70B",
		},
		{
			Keywords:     []string{"llama", "34b", "33b"},
			Capabilities: []string{"completion", "tools"},
			Family:       "llama",
			SizeHint:     65 * 1024 * 1024 * 1024, // 65GB
			ParamSize:    "34B",
		},
		{
			Keywords:     []string{"llama", "13b"},
			Capabilities: []string{"completion", "tools"},
			Family:       "llama",
			SizeHint:     26 * 1024 * 1024 * 1024, // 26GB
			ParamSize:    "13B",
		},
		{
			Keywords:     []string{"llama", "7b", "8b"},
			Capabilities: []string{"completion", "tools"},
			Family:       "llama",
			SizeHint:     14 * 1024 * 1024 * 1024, // 14GB
			ParamSize:    "7B",
		},
	}

	// GLM 模式
	mpm.patterns["glm"] = []ModelPattern{
		{
			Keywords:     []string{"glm"},
			Capabilities: []string{"completion", "tools"},
			Family:       "glm",
			SizeHint:     6 * 1024 * 1024 * 1024, // 默认6GB
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"glm", "4.6"},
			Capabilities: []string{"completion", "tools"},
			Family:       "glm",
			SizeHint:     10 * 1024 * 1024 * 1024, // 10GB
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"glm", "4"},
			Capabilities: []string{"completion", "tools"},
			Family:       "glm",
			SizeHint:     8 * 1024 * 1024 * 1024, // 8GB
			ParamSize:    "unknown",
		},
	}

	// 特殊功能模式
	mpm.patterns["special"] = []ModelPattern{
		{
			Keywords:     []string{"embedding"},
			Capabilities: []string{"embedding"},
			Family:       "embedding",
			SizeHint:     4 * 1024 * 1024 * 1024,
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"thinking", "o1"},
			Capabilities: []string{"completion", "thinking"},
			Family:       "thinking",
			SizeHint:     4 * 1024 * 1024 * 1024,
			ParamSize:    "unknown",
		},
		{
			Keywords:     []string{"vision"},
			Capabilities: []string{"vision"},
			Family:       "vision",
			SizeHint:     4 * 1024 * 1024 * 1024,
			ParamSize:    "unknown",
		},
	}

	// 其他模型家族
	mpm.patterns["mistral"] = []ModelPattern{
		{
			Keywords:     []string{"mistral"},
			Capabilities: []string{"completion", "tools"},
			Family:       "mistral",
			SizeHint:     4 * 1024 * 1024 * 1024,
			ParamSize:    "unknown",
		},
	}

	mpm.patterns["codellama"] = []ModelPattern{
		{
			Keywords:     []string{"codellama"},
			Capabilities: []string{"completion", "tools"},
			Family:       "codellama",
			SizeHint:     4 * 1024 * 1024 * 1024,
			ParamSize:    "unknown",
		},
	}

	mpm.patterns["qwen"] = []ModelPattern{
		{
			Keywords:     []string{"qwen"},
			Capabilities: []string{"completion", "tools"},
			Family:       "qwen",
			SizeHint:     4 * 1024 * 1024 * 1024,
			ParamSize:    "unknown",
		},
	}
}

// MatchModel 匹配模型并返回最佳匹配结果
func (mpm *ModelPatternMatcher) MatchModel(modelID string) *ModelMatchResult {
	modelID = strings.ToLower(modelID)

	var bestMatch *ModelMatchResult
	var highestScore = 0

	mpm.mu.RLock()
	defer mpm.mu.RUnlock()

	// 遍历所有模式寻找最佳匹配
	for category, patterns := range mpm.patterns {
		for _, pattern := range patterns {
			score := mpm.calculateMatchScore(modelID, pattern.Keywords)
			if score > highestScore {
				highestScore = score
				bestMatch = &ModelMatchResult{
					Category:     category,
					Capabilities: pattern.Capabilities,
					Family:       pattern.Family,
					SizeHint:     pattern.SizeHint,
					ParamSize:    pattern.ParamSize,
					MatchScore:   score,
				}
			}
		}
	}

	// 如果没有匹配，返回默认结果
	if bestMatch == nil {
		bestMatch = &ModelMatchResult{
			Category:     "unknown",
			Capabilities: []string{"completion"},
			Family:       "unknown",
			SizeHint:     4 * 1024 * 1024 * 1024, // 4GB 默认
			ParamSize:    "unknown",
			MatchScore:   0,
		}
	}

	return bestMatch
}

// calculateMatchScore 计算匹配分数
func (mpm *ModelPatternMatcher) calculateMatchScore(modelID string, keywords []string) int {
	score := 0
	for _, keyword := range keywords {
		if strings.Contains(modelID, keyword) {
			score += len(keyword) // 更长的关键词获得更高分数
		}
	}
	return score
}

// ModelMatchResult 模型匹配结果
type ModelMatchResult struct {
	Category     string
	Capabilities []string
	Family       string
	SizeHint     int64
	ParamSize    string
	MatchScore   int
}

// 全局模式匹配器实例
var globalPatternMatcher *ModelPatternMatcher
var patternMatcherOnce sync.Once

// GetGlobalPatternMatcher 获取全局模式匹配器（单例模式）
func GetGlobalPatternMatcher() *ModelPatternMatcher {
	patternMatcherOnce.Do(func() {
		globalPatternMatcher = NewModelPatternMatcher()
		globalPatternMatcher.InitializePatterns()
	})
	return globalPatternMatcher
}

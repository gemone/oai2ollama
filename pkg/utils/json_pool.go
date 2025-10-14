package utils

import (
	"bytes"
	"encoding/json"
	"sync"
)

// JSONPool 提供JSON序列化的对象池
type JSONPool struct {
	bufferPool sync.Pool
}

// NewJSONPool 创建新的JSON池
func NewJSONPool() *JSONPool {
	return &JSONPool{
		bufferPool: sync.Pool{
			New: func() interface{} {
				return new(bytes.Buffer)
			},
		},
	}
}

// Marshal 优化的JSON序列化
func (p *JSONPool) Marshal(v interface{}) ([]byte, error) {
	buf := p.bufferPool.Get().(*bytes.Buffer)
	defer func() {
		buf.Reset()
		p.bufferPool.Put(buf)
	}()

	encoder := json.NewEncoder(buf)
	encoder.SetEscapeHTML(false) // 避免HTML转义，提升性能

	if err := encoder.Encode(v); err != nil {
		return nil, err
	}

	// 移除末尾的换行符（json.Encode会自动添加）
	result := make([]byte, buf.Len()-1)
	copy(result, buf.Bytes()[:buf.Len()-1])
	return result, nil
}

// MarshalToString 优化的JSON序列化为字符串
func (p *JSONPool) MarshalToString(v interface{}) (string, error) {
	data, err := p.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// MustMarshal 不会返回错误的JSON序列化（用于兼容性）
func (p *JSONPool) MustMarshal(v interface{}) []byte {
	data, err := p.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return data
}

// Unmarshal 优化的JSON反序列化
func (p *JSONPool) Unmarshal(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}

// 全局JSON池实例
var GlobalJSONPool = NewJSONPool()

// 便捷函数
func MarshalJSON(v interface{}) ([]byte, error) {
	return GlobalJSONPool.Marshal(v)
}

func MarshalJSONToString(v interface{}) (string, error) {
	return GlobalJSONPool.MarshalToString(v)
}

func MustMarshalJSON(v interface{}) []byte {
	return GlobalJSONPool.MustMarshal(v)
}

func UnmarshalJSON(data []byte, v interface{}) error {
	return GlobalJSONPool.Unmarshal(data, v)
}

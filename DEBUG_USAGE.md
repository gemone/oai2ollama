# Debug模式使用说明

## 概述

oai2ollama 支持在debug模式下打印所有API接口的请求体和返回体，便于开发和调试。该功能使用了装饰器模式，代码更加整洁，并提供了完整的请求/响应跟踪。

## 启用Debug模式

### 方法1：环境变量

```bash
# 启用API调用调试
export DEBUG_API_CALLS=true

# 启用响应体日志打印（可选，默认为true）
export DEBUG_LOG_RESPONSE_BODY=true

# 启用thinking支持（可选）
export THINKING_ENABLE=true

# 运行应用
python -m oai2ollama
```

### 方法2：启动时设置

```bash
DEBUG_API_CALLS=true DEBUG_LOG_RESPONSE_BODY=true python -m oai2ollama
```

### 方法3：命令行参数

```bash
python -m oai2ollama --debug-api-calls true --debug-log-response-body true
```

## Debug输出格式

当debug模式启用时，每个API调用都会输出如下格式的日志：

```
=== API Call Debug ===
Method: POST
URL: /api/chat
Request Body:
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
Response Body:
{
  "model": "gpt-3.5-turbo",
  "created_at": "2025-10-12T16:45:40.442-08:00",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you?"
  },
  "done": true
}
=====================
```

### 响应摘要模式

当 `DEBUG_LOG_RESPONSE_BODY=false` 时，响应部分将显示摘要而非完整内容：

```
=== API Call Debug ===
Method: POST
URL: /api/chat
Request Body:
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
Response Summary:
{
  "status": "success",
  "data_keys": ["model", "created_at", "message", "done"]
}
=====================
```

## 配置选项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DEBUG_API_CALLS` | `false` | 是否启用API调用调试 |
| `DEBUG_LOG_RESPONSE_BODY` | `true` | 是否在debug日志中包含响应体（可能很大） |
| `THINKING_ENABLE` | `false` | 是否启用thinking支持 |

## 支持的端点

以下API端点都支持debug模式：

- `GET /api/tags` - 获取模型列表
- `POST /api/show` - 显示模型信息
- `GET /v1/models` - OpenAI格式模型列表
- `POST /v1/chat/completions` - OpenAI聊天完成
- `POST /api/generate` - Ollama文本生成
- `POST /api/chat` - Ollama聊天
- `POST /api/embeddings` - 嵌入向量生成
- `GET /api/ps` - 运行中的模型
- `POST /api/pull` - 拉取模型
- `DELETE /api/delete` - 删除模型
- `POST /api/copy` - 复制模型
- `POST /api/create` - 创建模型
- `GET /api/version` - Ollama版本

## 特殊功能

### 1. 请求转换显示
对于Ollama到OpenAI的转换接口（如`/api/generate`、`/api/chat`），debug日志会显示：
- 原始Ollama请求格式
- 转换后的OpenAI请求格式

```
=== API Call Debug ===
Method: POST
URL: http://localhost:11434/chat/completions
Request Body:
{
  "original_ollama_request": {
    "model": "llama2",
    "prompt": "Hello",
    "stream": false
  },
  "converted_openai_request": {
    "model": "llama2",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": false
  }
}
Response Body:
{
  "choices": [
    {
      "message": {
        "content": "Hi there!"
      }
    }
  ]
}
=====================
```

### 2. 错误日志
如果API调用失败，debug日志会显示错误信息而不是响应体。

### 3. 装饰器实现
使用了Python装饰器模式，自动处理：
- 请求体解析
- 响应体记录
- 错误捕获和记录
- 端点路径自动识别

## 使用场景

1. **调试连接问题** - 查看实际调用的 URL 是否正确
2. **验证请求格式** - 确认发送给后端 API 的请求格式
3. **监控 API 调用** - 跟踪实际的网络请求和响应
4. **性能分析** - 观察请求和响应的大小
5. **格式转换调试** - 查看Ollama到OpenAI格式的转换过程

## 示例

### 启用debug并运行服务

```bash
DEBUG_API_CALLS=true DEBUG_LOG_RESPONSE_BODY=true python -m oai2ollama
```

### 测试API调用

```bash
# 测试简单端点
curl http://localhost:11434/api/version

# 测试带请求体的端点
curl -X POST http://localhost:11434/api/show \
  -H "Content-Type: application/json" \
  -d '{"name": "test-model"}'

# 测试聊天接口
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

每个调用都会在控制台输出详细的debug信息。

## 注意事项

- Debug模式会产生额外的日志输出，可能影响性能
- `DEBUG_LOG_RESPONSE_BODY=false` 可以减少日志量，提高性能
- 建议仅在开发和调试时启用debug模式
- 调试信息包含敏感数据（API密钥、请求内容等），在生产环境中请谨慎使用
- 流式请求会记录请求开始，但不会记录每个流式响应块以避免日志过多
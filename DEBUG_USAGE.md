# API 调用调试功能使用说明

## 概述

oai2ollama 现在支持在运行时查看实际的 API 调用地址和内容，这对于调试和监控很有帮助。

## 环境变量配置

### 启用调试模式
```bash
export DEBUG_API_CALLS=true
```

### 控制是否记录响应体（可选）
```bash
export DEBUG_LOG_RESPONSE_BODY=true   # 记录完整响应（默认）
export DEBUG_LOG_RESPONSE_BODY=false  # 仅记录响应摘要
```

## 命令行选项

### 启用调试模式
```bash
python -m oai2ollama --debug-api-calls true
```

### 控制响应体记录
```bash
python -m oai2ollama --debug-api-calls true --debug-log-response-body false
```

## 调试输出格式

启用调试后，每次 API 调用都会输出类似以下格式的信息：

```
=== API Call Debug ===
Method: POST
URL: http://localhost:11434/chat/completions
Request: {
  "model": "llama2",
  "messages": [
    {
      "role": "user",
      "content": "hello"
    }
  ]
}
Response: {
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

### 响应摘要模式

当 `DEBUG_LOG_RESPONSE_BODY=false` 时，响应部分将显示摘要而非完整内容：

```
=== API Call Debug ===
Method: POST
URL: http://localhost:11434/chat/completions
Request: {
  "model": "llama2",
  "messages": [
    {
      "role": "user",
      "content": "hello"
    }
  ]
}
Response: {
  "status": "success",
  "data_keys": [
    "choices"
  ]
}
=====================
```

## 使用场景

1. **调试连接问题** - 查看实际调用的 URL 是否正确
2. **验证请求格式** - 确认发送给后端 API 的请求格式
3. **监控 API 调用** - 跟踪实际的网络请求和响应
4. **性能分析** - 观察请求和响应的大小

## 注意事项

- 调试信息包含敏感数据（API 密钥、请求内容等），在生产环境中请谨慎使用
- 响应体可能很大，在处理大量数据时建议使用摘要模式
- 流式请求只会记录请求信息，不会记录流式响应内容
import json
import logging
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, Response
from typing import Optional

from .config import env
from .database import db_manager
from .metrics import track_api_metrics

app = FastAPI()


@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "oai2ollama",
        "description": "OpenAI to Ollama API compatibility layer",
        "version": "1.2.6",
        "endpoints": {
            "ollama_api": {
                "generate": "POST /api/generate",
                "chat": "POST /api/chat",
                "embeddings": "POST /api/embeddings",
                "tags": "GET /api/tags",
                "show": "POST /api/show",
                "ps": "GET /api/ps",
                "pull": "POST /api/pull",
                "delete": "DELETE /api/delete",
                "copy": "POST /api/copy",
                "create": "POST /api/create",
                "version": "GET /api/version",
            },
            "openai_api": {"models": "GET /v1/models", "chat_completions": "POST /v1/chat/completions"},
            "metrics": {"metrics": "GET /metrics or GET /metric"},
        },
    }


@app.head("/")
async def root_head():
    """HEAD endpoint for root path"""
    return Response(status_code=200)


@app.get("/metrics")
@app.get("/metric")
async def get_metrics(model: Optional[str] = Query(None, description="Filter by specific model"), days: int = Query(7, ge=1, le=365, description="Number of days for daily stats"), limit: int = Query(100, ge=1, le=1000, description="Limit for recent calls")):
    """Get API usage metrics and statistics"""
    try:
        # Get overall statistics
        overall_stats = db_manager.get_overall_stats()

        # Get model statistics
        model_stats = db_manager.get_model_stats(model)

        # Get daily statistics
        daily_stats = db_manager.get_daily_stats(days)

        # Get recent calls
        recent_calls = db_manager.get_recent_calls(limit, model)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall": overall_stats,
            "models": model_stats,
            "daily": daily_stats,
            "recent_calls": recent_calls[:10],  # Only return first 10 recent calls
            "filters": {"model": model, "days": days, "limit": limit},
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"status": "error", "message": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


def get_ollama_timestamp():
    """Generate Ollama-compatible timestamp with timezone offset"""
    now = datetime.now(timezone(timedelta(hours=-7)))  # PST timezone (-7:00)
    # Format: 2023-08-04T08:52:19.385-07:00 (note the colon in timezone)
    timezone_str = now.strftime("%z")
    timezone_with_colon = timezone_str[:3] + ":" + timezone_str[3:]
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + timezone_with_colon


# 设置调试日志
logger = logging.getLogger(__name__)
if env.debug_api_calls:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _log_api_call(method: str, url: str, request_data: dict = None, response_data: dict = None, error: str = None, debug_config=None):
    """记录API调用信息"""
    # 使用传入的配置或全局配置
    debug_enabled = getattr(debug_config, "debug_api_calls", env.debug_api_calls) if debug_config else env.debug_api_calls
    log_response_body = getattr(debug_config, "debug_log_response_body", env.debug_log_response_body) if debug_config else env.debug_log_response_body

    if not debug_enabled:
        return

    print(f"\n=== API Call Debug ===")
    print(f"Method: {method}")
    print(f"URL: {url}")
    if request_data:
        print(f"Request Body:")
        print(f"{json.dumps(request_data, indent=2, ensure_ascii=False)}")
    if response_data and log_response_body:
        print(f"Response Body:")
        print(f"{json.dumps(response_data, indent=2, ensure_ascii=False)}")
    elif response_data:
        response_summary = {"status": "success", "data_keys": list(response_data.keys()) if isinstance(response_data, dict) else "non-dict response"}
        print(f"Response Summary: {json.dumps(response_summary, indent=2, ensure_ascii=False)}")
    if error:
        print(f"Error: {error}")
    print(f"=====================\n")


def debug_api(endpoint: str = None, method: str = "GET"):
    """
    API调试装饰器

    Args:
        endpoint: API端点路径，如果为None则从函数路径推断
        method: HTTP方法
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not env.debug_api_calls:
                return await func(*args, **kwargs)

            # 确定端点路径
            actual_endpoint = endpoint
            if actual_endpoint is None:
                # 从函数路径推断端点
                func_name = func.__name__
                if func_name == "models":
                    actual_endpoint = "/api/tags"
                elif func_name == "list_models":
                    actual_endpoint = "/v1/models"
                elif func_name == "chat_completions":
                    actual_endpoint = "/v1/chat/completions"
                elif func_name == "ollama_chat":
                    actual_endpoint = "/api/chat"
                elif func_name == "generate_text":
                    actual_endpoint = "/api/generate"
                elif func_name == "create_embeddings":
                    actual_endpoint = "/api/embeddings"
                elif func_name == "show_model":
                    actual_endpoint = "/api/show"
                elif func_name == "list_running_models":
                    actual_endpoint = "/api/ps"
                elif func_name == "pull_model":
                    actual_endpoint = "/api/pull"
                elif func_name == "delete_model":
                    actual_endpoint = "/api/delete"
                elif func_name == "copy_model":
                    actual_endpoint = "/api/copy"
                elif func_name == "create_model":
                    actual_endpoint = "/api/create"
                elif func_name == "ollama_version":
                    actual_endpoint = "/api/version"
                else:
                    actual_endpoint = f"/unknown/{func_name}"

            # 尝试从参数中提取请求数据
            request_data = None
            response_data = None
            error = None

            try:
                result = await func(*args, **kwargs)
                response_data = result

                # 尝试从第一个参数获取Request对象并提取JSON数据
                if args and hasattr(args[0], "json"):
                    try:
                        request_data = await args[0].json()
                    except:
                        pass

                # Check if response is a streaming response
                from fastapi.responses import StreamingResponse

                if isinstance(result, StreamingResponse):
                    # For streaming responses, don't try to serialize the response body
                    _log_api_call(method, actual_endpoint, request_data=request_data, response_data=None)
                else:
                    _log_api_call(method, actual_endpoint, request_data=request_data, response_data=response_data)
                return result

            except Exception as e:
                error = str(e)
                # 尝试从第一个参数获取Request对象并提取JSON数据
                if args and hasattr(args[0], "json"):
                    try:
                        request_data = await args[0].json()
                    except:
                        pass

                _log_api_call(method, actual_endpoint, request_data=request_data, error=error)
                raise

        return wrapper

    return decorator


def _new_client(base_url: str | None = None):
    from httpx import AsyncClient

    url = base_url or str(env.base_url)
    return AsyncClient(base_url=url, headers={"Authorization": f"Bearer {env.api_key}"}, timeout=60, http2=True, follow_redirects=True)


def _new_streaming_client(base_url: str | None = None):
    from httpx import AsyncClient, Timeout

    url = base_url or str(env.base_url)
    # For streaming requests, use no timeout to allow long-running responses
    return AsyncClient(base_url=url, headers={"Authorization": f"Bearer {env.api_key}"}, timeout=None, http2=True, follow_redirects=True)


@app.get("/api/tags")
@debug_api(method="GET")
async def models():
    models_map = {}

    # 先添加额外配置的模型作为基础
    for model in env.extra_models:
        # 生成正确的时间格式，包含时区偏移
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-8)))
        modified_at = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "-08:00"

        models_map[model] = {
            "name": model,
            "modified_at": modified_at,
            "size": 7365960935,  # 合理的大小
            "digest": f"sha256:{hash(model) % 1000000000000000000:x}",
            "details": {"format": "gguf", "family": "llama", "families": None, "parameter_size": "13B", "quantization_level": "Q4_0"},
        }

    # 尝试从 fetch_model_url 获取模型列表
    fetch_urls = []
    if env.fetch_model_url:
        fetch_urls.append(str(env.fetch_model_url))
    fetch_urls.append(str(env.base_url))

    for url in fetch_urls:
        full_url = f"{url.rstrip('/')}/models"
        try:
            async with _new_client(url) as client:
                res = await client.get("/models")
                res.raise_for_status()
                models_data = res.json().get("data", [])

                for i in models_data:
                    model_id = i["id"]
                    # 生成正确的时间格式，包含时区偏移
                    from datetime import datetime, timezone, timedelta

                    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-8)))
                    modified_at = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "-08:00"

                    models_map[model_id] = {"name": model_id, "model": model_id, "modified_at": modified_at, "size": 7365960935, "digest": f"sha256:{hash(model_id) % 1000000000000000000:x}", "details": {"format": "gguf", "family": "llama", "families": None, "parameter_size": "13B", "quantization_level": "Q4_0"}}

                # 记录成功的API调用
                _log_api_call("GET", full_url, response_data=models_data)
                break  # 成功获取则跳出循环
        except Exception as e:
            # 记录失败的API调用
            _log_api_call("GET", full_url, error=str(e))

            # 如果是最后一个URL也失败了，则使用现有的模型列表
            if url == fetch_urls[-1]:
                print(f"Failed to fetch models from {url}: {e}, using existing models only")

    return {"models": list(models_map.values())}


@app.post("/api/show")
@debug_api(method="POST")
async def show_model(request: Request):
    return {
        "model_info": {"general.architecture": "CausalLM"},
        "capabilities": ["completion", *env.capabilities],
    }


@app.get("/v1/models")
@debug_api(method="GET")
async def list_models():
    # 尝试从不同的URL获取模型列表
    fetch_urls = []
    if env.fetch_model_url:
        fetch_urls.append(str(env.fetch_model_url))
    fetch_urls.append(str(env.base_url))

    for url in fetch_urls:
        full_url = f"{url.rstrip('/')}/models"
        try:
            async with _new_client(url) as client:
                res = await client.get("/models")
                res.raise_for_status()
                response_data = res.json()

                # 记录成功的API调用
                _log_api_call("GET", full_url, response_data=response_data)
                return response_data
        except Exception as e:
            # 记录失败的API调用
            _log_api_call("GET", full_url, error=str(e))

            # 如果是最后一个URL也失败了，返回只包含额外模型的响应
            if url == fetch_urls[-1]:
                print(f"Failed to fetch models from {url}: {e}, returning extra models only")
                return {"data": [{"id": model, "object": "model"} for model in env.extra_models]}

    # 如果所有URL都失败，返回空列表
    return {"data": []}


@app.post("/v1/chat/completions")
@debug_api(method="POST")
@track_api_metrics("/v1/chat/completions", "POST")
async def chat_completions(request: Request):
    data = await request.json()
    full_url = f"{str(env.base_url).rstrip('/')}/chat/completions"

    # 处理 thinking 支持
    if env.thinking_enable and "thinking" not in data:
        data["thinking"] = {"type": "enabled"}

    if data.get("stream", False):

        async def stream():
            async with _new_streaming_client() as client, client.stream("POST", "/chat/completions", json=data) as response:
                # 记录流式请求
                _log_api_call("POST", full_url, request_data={**data, "stream": True})

                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(stream(), media_type="text/event-stream")

    else:
        try:
            async with _new_client() as client:
                res = await client.post("/chat/completions", json=data)
                res.raise_for_status()
                response_data = res.json()

                # 记录非流式请求和响应
                _log_api_call("POST", full_url, request_data=data, response_data=response_data)
                return response_data
        except Exception as e:
            # 记录失败的请求
            _log_api_call("POST", full_url, request_data=data, error=str(e))
            raise


@app.post("/api/generate")
@debug_api(method="POST")
@track_api_metrics("/api/generate", "POST")
async def generate_text(request: Request):
    """Generate text completion using OpenAI-compatible API"""
    data = await request.json()
    full_url = f"{str(env.base_url).rstrip('/')}/chat/completions"

    # Convert Ollama format to OpenAI format
    options = data.get("options") or {}
    prompt = data.get("prompt", "")
    model = data.get("model", "gpt-3.5-turbo")

    # Convert prompt to messages format
    # Handle empty prompt by providing a default
    content = prompt if prompt.strip() else "Hello"

    openai_request = {"model": model, "messages": [{"role": "user", "content": content}]}

    # Only add optional parameters if they exist
    if options.get("num_predict"):
        openai_request["max_tokens"] = options.get("num_predict")
    if options.get("temperature") is not None:
        openai_request["temperature"] = options.get("temperature")
    if data.get("stream"):
        openai_request["stream"] = True

    # Debug logging - 记录原始Ollama请求和转换后的OpenAI请求
    debug_data = {"original_ollama_request": data, "converted_openai_request": openai_request}
    _log_api_call("POST", full_url, request_data=debug_data)

    if data.get("stream", False):

        async def stream():
            try:
                async with _new_streaming_client() as client, client.stream("POST", "/chat/completions", json=openai_request) as response:
                    _log_api_call("POST", full_url, request_data=openai_request)

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                chunk_data = json.loads(line[6:])
                                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    text = delta.get("content", "")
                                    if text:
                                        # Convert to Ollama streaming format
                                        ollama_response = {"model": data.get("model"), "created_at": get_ollama_timestamp(), "response": text, "done": False}
                                        yield json.dumps(ollama_response) + "\n"
                            except json.JSONDecodeError:
                                continue
                        elif line.strip() == "data: [DONE]":
                            # Final response
                            final_response = {"model": data.get("model"), "created_at": get_ollama_timestamp(), "response": "", "done": True, "total_duration": 0, "prompt_eval_count": 0, "prompt_eval_duration": 0, "eval_count": 0, "eval_duration": 0}
                            yield json.dumps(final_response) + "\n"
            except Exception as e:
                # 记录错误并返回错误响应
                _log_api_call("POST", full_url, request_data=openai_request, error=str(e))
                error_msg = f"Error: {str(e)}"
                error_response = {"model": data.get("model"), "created_at": get_ollama_timestamp(), "response": error_msg, "done": True, "error": error_msg}
                yield json.dumps(error_response) + "\n"

        return StreamingResponse(stream(), media_type="text/x-ndjson")
    else:
        try:
            async with _new_client() as client:
                res = await client.post("/chat/completions", json=openai_request)
                res.raise_for_status()
                response_data = res.json()

                _log_api_call("POST", full_url, request_data=openai_request, response_data=response_data)

                # Convert OpenAI response to Ollama format
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    message = response_data["choices"][0].get("message", {})
                    text = message.get("content", "")
                    return {"model": data.get("model"), "created_at": get_ollama_timestamp(), "response": text, "done": True, "total_duration": 0, "prompt_eval_count": 0, "prompt_eval_duration": 0, "eval_count": 0, "eval_duration": 0}
                else:
                    return {"model": model, "created_at": get_ollama_timestamp(), "response": "No response generated", "done": True, "error": "No response generated"}
        except Exception as e:
            _log_api_call("POST", full_url, request_data=openai_request, error=str(e))

            # Return error in Ollama format instead of raising
            error_msg = f"Error: {str(e)}"
            return {"model": model, "created_at": get_ollama_timestamp(), "response": error_msg, "done": True, "error": error_msg}


@app.post("/api/chat")
@debug_api(method="POST")
@track_api_metrics("/api/chat", "POST")
async def ollama_chat(request: Request):
    """Ollama chat endpoint using OpenAI-compatible API"""
    data = await request.json()
    full_url = f"{str(env.base_url).rstrip('/')}/chat/completions"

    # Convert Ollama format to OpenAI format
    options = data.get("options") or {}
    openai_request = {"model": data.get("model", "gpt-3.5-turbo"), "messages": data.get("messages", []), "max_tokens": options.get("num_predict", 500), "temperature": options.get("temperature", 0.7), "stream": data.get("stream", False)}

    # Add thinking support if enabled
    if env.thinking_enable and "thinking" not in openai_request:
        openai_request["thinking"] = {"type": "enabled"}

    # Debug logging - 记录原始Ollama请求和转换后的OpenAI请求
    debug_data = {"original_ollama_request": data, "converted_openai_request": openai_request}

    if data.get("stream", False):

        async def stream():
            try:
                async with _new_streaming_client() as client, client.stream("POST", "/chat/completions", json=openai_request) as response:
                    _log_api_call("POST", full_url, request_data={**debug_data, "stream": True})

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                chunk_data = json.loads(line[6:])
                                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        # Convert to Ollama streaming format
                                        ollama_response = {"model": data.get("model"), "created_at": get_ollama_timestamp(), "message": {"role": "assistant", "content": delta["content"]}, "done": False}
                                        yield json.dumps(ollama_response) + "\n"
                            except json.JSONDecodeError:
                                continue
                        elif line.strip() == "data: [DONE]":
                            # Final response
                            final_response = {"model": data.get("model"), "created_at": get_ollama_timestamp(), "message": {"role": "assistant", "content": ""}, "done": True, "total_duration": 0, "prompt_eval_count": 0, "prompt_eval_duration": 0, "eval_count": 0, "eval_duration": 0}
                            yield json.dumps(final_response) + "\n"
            except Exception as e:
                # 记录错误并返回错误响应
                _log_api_call("POST", full_url, request_data=openai_request, error=str(e))
                error_response = {"model": data.get("model"), "created_at": get_ollama_timestamp(), "message": {"role": "assistant", "content": f"Error: {str(e)}"}, "done": True, "error": f"Error: {str(e)}"}
                yield json.dumps(error_response) + "\n"

        return StreamingResponse(stream(), media_type="text/x-ndjson")
    else:
        try:
            async with _new_client() as client:
                res = await client.post("/chat/completions", json=openai_request)
                res.raise_for_status()
                response_data = res.json()

                _log_api_call("POST", full_url, request_data=openai_request, response_data=response_data)

                # Convert OpenAI response to Ollama format
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    message = response_data["choices"][0].get("message", {})
                    return {"model": data.get("model"), "created_at": get_ollama_timestamp(), "message": {"role": message.get("role", "assistant"), "content": message.get("content", "")}, "done": True, "total_duration": 0, "prompt_eval_count": 0, "prompt_eval_duration": 0, "eval_count": 0, "eval_duration": 0}
                else:
                    return {"error": "No response generated"}
        except Exception as e:
            _log_api_call("POST", full_url, request_data=openai_request, error=str(e))
            raise


@app.post("/api/embeddings")
@debug_api(method="POST")
@track_api_metrics("/api/embeddings", "POST")
async def create_embeddings(request: Request):
    """Create embeddings using OpenAI-compatible API"""
    data = await request.json()
    full_url = f"{str(env.base_url).rstrip('/')}/embeddings"

    # Convert Ollama format to OpenAI format
    openai_request = {"model": data.get("model", "text-embedding-ada-002"), "input": data.get("prompt", "")}

    try:
        async with _new_client() as client:
            res = await client.post("/embeddings", json=openai_request)
            res.raise_for_status()
            response_data = res.json()

            _log_api_call("POST", full_url, request_data=openai_request, response_data=response_data)

            # Convert OpenAI response to Ollama format
            if "data" in response_data and len(response_data["data"]) > 0:
                embedding = response_data["data"][0].get("embedding", [])
                return {"embeddings": [embedding]}
            else:
                return {"error": "No embedding generated"}
    except Exception as e:
        _log_api_call("POST", full_url, request_data=openai_request, error=str(e))
        # If embeddings endpoint doesn't exist, return a mock response
        return {
            "embeddings": [[0.0] * 1536]  # Default embedding size
        }


@app.get("/api/ps")
@debug_api(method="GET")
async def list_running_models():
    """List running models - mock implementation"""
    # Since we're proxying to OpenAI API, we don't have actual running models info
    # Return empty list or mock data
    return {"models": []}


@app.post("/api/pull")
@debug_api(method="POST")
async def pull_model(request: Request):
    """Pull model - mock implementation since we're proxying"""
    data = await request.json()
    model_name = data.get("name", "")

    # Mock response since we can't actually pull models through OpenAI API
    return {"status": "pulling model", "digest": f"sha256:{hash(model_name) % 1000000000000000000:x}"}


@app.delete("/api/delete")
@debug_api(method="DELETE")
async def delete_model(request: Request):
    """Delete model - mock implementation since we're proxying"""
    data = await request.json()
    model_name = data.get("name", "")

    # Mock response since we can't actually delete models through OpenAI API
    return {"status": "success"}


@app.post("/api/copy")
@debug_api(method="POST")
async def copy_model(request: Request):
    """Copy model - mock implementation since we're proxying"""
    data = await request.json()
    source = data.get("source", "")
    destination = data.get("destination", "")

    # Mock response since we can't actually copy models through OpenAI API
    return {"status": "success"}


@app.post("/api/create")
@debug_api(method="POST")
async def create_model(request: Request):
    """Create model - mock implementation since we're proxying"""
    data = await request.json()
    model_name = data.get("name", "")

    # Mock response since we can't actually create models through OpenAI API
    return {"status": "creating model", "digest": f"sha256:{hash(model_name) % 1000000000000000000:x}"}


@app.get("/api/version")
@debug_api(method="GET")
async def ollama_version():
    return {"version": "0.11.4"}

import json
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from .config import env

app = FastAPI()

# 设置调试日志
logger = logging.getLogger(__name__)
if env.debug_api_calls:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _log_api_call(method: str, url: str, request_data: dict = None, response_data: dict = None, error: str = None, debug_config=None):
    """记录API调用信息"""
    # 使用传入的配置或全局配置
    debug_enabled = getattr(debug_config, 'debug_api_calls', env.debug_api_calls) if debug_config else env.debug_api_calls
    log_response_body = getattr(debug_config, 'debug_log_response_body', env.debug_log_response_body) if debug_config else env.debug_log_response_body

    if not debug_enabled:
        return

    print(f"\n=== API Call Debug ===")
    print(f"Method: {method}")
    print(f"URL: {url}")
    if request_data:
        print(f"Request: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
    if response_data and log_response_body:
        print(f"Response: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    elif response_data:
        response_summary = {"status": "success", "data_keys": list(response_data.keys()) if isinstance(response_data, dict) else "non-dict response"}
        print(f"Response: {json.dumps(response_summary, indent=2, ensure_ascii=False)}")
    if error:
        print(f"Error: {error}")
    print(f"=====================\n")


def _new_client(base_url: str | None = None):
    from httpx import AsyncClient

    url = base_url or str(env.base_url)
    return AsyncClient(base_url=url, headers={"Authorization": f"Bearer {env.api_key}"}, timeout=60, http2=True, follow_redirects=True)


@app.get("/api/tags")
async def models():
    models_map = {}

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
                models_map.update({i["id"]: {"name": i["id"], "model": i["id"]} for i in models_data})

                # 记录成功的API调用
                _log_api_call("GET", full_url, response_data=models_data)
                break  # 成功获取则跳出循环
        except Exception as e:
            # 记录失败的API调用
            _log_api_call("GET", full_url, error=str(e))

            # 如果是最后一个URL也失败了，则使用额外的模型列表
            if url == fetch_urls[-1]:
                print(f"Failed to fetch models from {url}: {e}, using extra models only")

    # 添加额外配置的模型
    models_map.update({i: {"name": i, "model": i} for i in env.extra_models})

    return {"models": list(models_map.values())}


@app.post("/api/show")
async def show_model():
    return {
        "model_info": {"general.architecture": "CausalLM"},
        "capabilities": ["completion", *env.capabilities],
    }


@app.get("/v1/models")
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
async def chat_completions(request: Request):
    data = await request.json()
    full_url = f"{str(env.base_url).rstrip('/')}/chat/completions"

    # 处理 thinking 支持
    if env.thinking_enable and "thinking" not in data:
        data["thinking"] = {"type": "enabled"}

    if data.get("stream", False):

        async def stream():
            async with _new_client() as client, client.stream("POST", "/chat/completions", json=data) as response:
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


@app.get("/api/version")
async def ollama_version():
    return {"version": "0.11.4"}

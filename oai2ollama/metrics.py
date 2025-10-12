import json
import logging
import time
from typing import Any

from .database import db_manager

logger = logging.getLogger(__name__)


class TokenCounter:
    """Token counter for extracting token counts from requests and responses"""

    @staticmethod
    def count_tokens_from_request(request_data: dict[str, object] | None, endpoint: str) -> tuple[int, int]:
        """
        从请求数据中计算token数量

        Returns:
            tuple: (request_tokens, estimated_response_tokens)
        """
        if not request_data:
            return 0, 0

        request_tokens = 0
        estimated_response_tokens = 0

        try:
            if endpoint in ["/v1/chat/completions", "/api/chat"]:
                # Chat completions - count messages
                messages = request_data.get("messages", [])
                if isinstance(messages, list):
                    for message in messages:
                        if isinstance(message, dict):
                            content = message.get("content", "")
                            if isinstance(content, str):
                                # Simple estimation: count tokens by characters/4 (English average)
                                # For Chinese text, character count is more accurate
                                if any("\u4e00" <= char <= "\u9fff" for char in content):
                                    request_tokens += len(content)  # 中文按字符数
                                else:
                                    request_tokens += len(content) // 4  # 英文按单词数估算

                            # 计算role的token
                            role = message.get("role", "")
                            if isinstance(role, str):
                                request_tokens += len(role) // 4

                # Estimate response tokens (based on max_tokens or default value)
                max_tokens = request_data.get("max_tokens", 1000)
                estimated_response_tokens = min(max_tokens, 1000) if isinstance(max_tokens, int) else 1000

            elif endpoint == "/api/generate":
                # Generate endpoint
                prompt = request_data.get("prompt", "")
                if isinstance(prompt, str):
                    if any("\u4e00" <= char <= "\u9fff" for char in prompt):
                        request_tokens += len(prompt)
                    else:
                        request_tokens += len(prompt) // 4

                # 估算响应token数
                options = request_data.get("options", {})
                if isinstance(options, dict):
                    num_predict = options.get("num_predict", 1000)
                    estimated_response_tokens = min(num_predict, 1000) if isinstance(num_predict, int) else 1000
                else:
                    estimated_response_tokens = 1000

            elif endpoint == "/api/embeddings":
                # Embeddings endpoint
                input_data = request_data.get("input", "")
                if isinstance(input_data, str):
                    if any("\u4e00" <= char <= "\u9fff" for char in input_data):
                        request_tokens += len(input_data)
                    else:
                        request_tokens += len(input_data) // 4
                elif isinstance(input_data, list):
                    for item in input_data:
                        if isinstance(item, str):
                            if any("\u4e00" <= char <= "\u9fff" for char in item):
                                request_tokens += len(item)
                            else:
                                request_tokens += len(item) // 4

                # Embeddings通常返回固定大小的向量
                estimated_response_tokens = 100

        except Exception as e:
            logger.warning(f"Error counting tokens from request: {e}")
            return 0, 100  # 默认值

        return request_tokens, estimated_response_tokens

    @staticmethod
    def count_tokens_from_response(response_data: Any, endpoint: str) -> int:
        """
        从响应数据中计算实际的token数量

        Returns:
            int: response_tokens
        """
        if not response_data:
            return 0

        response_tokens = 0

        try:
            if endpoint in ["/v1/chat/completions", "/api/chat"]:
                # Chat completions response
                if isinstance(response_data, dict):
                    choices = response_data.get("choices", [])
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                        if isinstance(choice, dict):
                            message = choice.get("message", {})
                            if isinstance(message, dict):
                                content = message.get("content", "")
                                if isinstance(content, str):
                                    if any("\u4e00" <= char <= "\u9fff" for char in content):
                                        response_tokens += len(content)
                                    else:
                                        response_tokens += len(content) // 4

                    # Also check usage field (if available)
                    usage = response_data.get("usage", {})
                    if isinstance(usage, dict):
                        completion_tokens = usage.get("completion_tokens")
                        if isinstance(completion_tokens, int):
                            response_tokens = completion_tokens

            elif endpoint == "/api/generate":
                # Generate response
                if isinstance(response_data, dict):
                    response = response_data.get("response", "")
                    if isinstance(response, str):
                        if any("\u4e00" <= char <= "\u9fff" for char in response):
                            response_tokens += len(response)
                        else:
                            response_tokens += len(response) // 4

            elif endpoint == "/api/embeddings" and isinstance(response_data, dict):
                embeddings = response_data.get("embeddings", [])
                if isinstance(embeddings, list):
                    # 每个embedding通常是固定长度的向量
                    response_tokens = len(embeddings) * 100  # 估算值

        except Exception as e:
            logger.warning(f"Error counting tokens from response: {e}")
            return 100  # 默认值

        return response_tokens


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.token_counter = TokenCounter()

    def record_api_call(self, endpoint: str, method: str, model: str | None = None, request_data: dict[str, object] | None = None, response_data: Any = None, status_code: int | None = None, response_time_ms: float | None = None, error_message: str | None = None):
        """记录API调用指标"""
        try:
            # 计算token数量
            request_tokens, estimated_response_tokens = self.token_counter.count_tokens_from_request(request_data, endpoint)

            # If actual response data exists, use actual token count
            response_tokens = self.token_counter.count_tokens_from_response(response_data, endpoint) if response_data and not error_message else estimated_response_tokens

            # 记录到数据库
            db_manager.record_api_call(
                endpoint=endpoint, method=method, model=model, request_tokens=request_tokens, response_tokens=response_tokens, status_code=status_code, response_time_ms=response_time_ms, error_message=error_message, request_data=request_data, metadata={"estimated_response_tokens": estimated_response_tokens}
            )

            logger.debug(f"Recorded metrics for {endpoint} - Model: {model}, Tokens: {request_tokens + response_tokens}")

        except Exception:
            logger.exception("Error recording API call metrics")


async def _wrap_streaming_response(streaming_response, endpoint: str, model: str | None, request_data: dict[str, object] | None, response_time_ms: float):
    """Wrap streaming response to collect actual content for token statistics"""
    from fastapi.responses import StreamingResponse

    collected_content = []

    async def content_generator():
        try:
            async for chunk in streaming_response.body_iterator:
                if chunk:
                    collected_content.append(chunk)
                    yield chunk
        except Exception:
            logging.exception("Error in streaming response")
            raise
        finally:
            # After stream ends, collect content statistics
            try:
                full_content = b"".join(collected_content).decode("utf-8")

                # Try to parse JSON response
                try:
                    if full_content.strip().startswith("data: "):
                        # Handle SSE format
                        lines = [line for line in full_content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]
                        content_parts = []
                        for line in lines:
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                if data.get("choices"):
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content_parts.append(delta["content"])
                                elif "response" in data:  # Ollama format
                                    content_parts.append(data["response"])
                            except json.JSONDecodeError:
                                continue

                        reconstructed_response = {"choices": [{"message": {"content": "".join(content_parts)}}]}
                    else:
                        # Try to parse directly as JSON
                        reconstructed_response = json.loads(full_content)

                except (json.JSONDecodeError, KeyError):
                    # If parsing fails, create a simple response structure
                    reconstructed_response = {"choices": [{"message": {"content": full_content}}]}

                # Record streaming response metrics
                metrics_collector.record_api_call(endpoint=endpoint, method="POST", model=model, request_data=request_data, response_data=reconstructed_response, status_code=200, response_time_ms=response_time_ms, error_message=None)

            except Exception:
                logging.exception("Error recording streaming metrics")

    # Create new StreamingResponse
    return StreamingResponse(content_generator(), status_code=streaming_response.status_code, headers=streaming_response.headers, media_type=streaming_response.media_type)


# 全局指标收集器实例
metrics_collector = MetricsCollector()


def track_api_metrics(endpoint: str, method: str = "POST"):
    """
    Decorator: Automatically track API call metrics

    Args:
        endpoint: API端点路径
        method: HTTP方法
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            request_data = None
            model = None
            response_data = None
            status_code = 200
            error_message = None

            try:
                # 尝试提取请求数据和模型信息
                if args and hasattr(args[0], "json"):
                    try:
                        request_data = await args[0].json()
                        # 提取模型名称
                        if endpoint in ["/v1/chat/completions", "/api/chat", "/api/generate", "/api/embeddings"]:
                            model = request_data.get("model") if request_data else None
                    except Exception:
                        pass

                # 执行原函数
                result = await func(*args, **kwargs)

                # 计算响应时间
                response_time_ms = (time.time() - start_time) * 1000

                # 处理不同类型的响应
                from fastapi.responses import StreamingResponse

                if isinstance(result, StreamingResponse):
                    # For streaming responses, we need to wrap them to collect actual content
                    response_data = None
                    wrapped_response = await _wrap_streaming_response(result, endpoint, model, request_data, response_time_ms)
                    return wrapped_response
                else:
                    response_data = result

                # Record metrics
                metrics_collector.record_api_call(endpoint=endpoint, method=method, model=model, request_data=request_data, response_data=response_data, status_code=status_code, response_time_ms=response_time_ms, error_message=error_message)

            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                error_message = str(e)
                status_code = 500

                # Record error metrics
                metrics_collector.record_api_call(endpoint=endpoint, method=method, model=model, request_data=request_data, response_data=None, status_code=status_code, response_time_ms=response_time_ms, error_message=error_message)

                raise
            else:
                return result

        return wrapper

    return decorator

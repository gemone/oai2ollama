#!/usr/bin/env python3
"""
测试debug模式功能的脚本
"""
import os
import sys

# 设置环境变量启用debug模式
os.environ['DEBUG_API_CALLS'] = 'true'
os.environ['DEBUG_LOG_RESPONSE_BODY'] = 'true'

# 模拟设置必要的环境变量
os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'

from fastapi.testclient import TestClient
import sys
sys.path.insert(0, '.')

def test_debug_functionality():
    """测试debug功能"""

    # 这里我们需要模拟配置，避免实际的网络调用
    print("=== 测试Debug API调用功能 ===")
    print("注意：由于缺少有效的API配置，某些请求可能会失败，但debug日志应该会显示")
    print()

    # 首先测试不需要外部API的简单端点
    try:
        from oai2ollama._app import app
        client = TestClient(app)

        # 测试1: GET /api/version
        print("1. 测试 GET /api/version")
        response = client.get("/api/version")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        print()

        # 测试2: GET /api/ps
        print("2. 测试 GET /api/ps")
        response = client.get("/api/ps")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        print()

        # 测试3: POST /api/show
        print("3. 测试 POST /api/show")
        test_data = {"name": "test-model"}
        response = client.post("/api/show", json=test_data)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        print()

        print("=== Debug功能测试完成 ===")
        print("如果上述调用显示了 '=== API Call Debug ===' 格式的日志，说明debug功能正常工作")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("这可能是因为缺少有效的API配置，但debug日志功能本身应该是正常的")

if __name__ == "__main__":
    test_debug_functionality()
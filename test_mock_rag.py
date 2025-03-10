#!/usr/bin/env python3
"""
简化版的RAG集成测试，只使用模拟模式。
"""

import os
import sys
import json
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# 导入必要的组件
from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.events import EventStream, EventSource
from openhands.storage.local import LocalFileStore

# 测试函数
def test_mock_rag():
    """测试RAG代码搜索集成的模拟版本。"""
    
    # 创建临时目录用于文件存储
    temp_dir = tempfile.TemporaryDirectory()
    file_store_path = temp_dir.name
    
    # 创建LocalFileStore
    file_store = LocalFileStore(root=file_store_path)
    
    # 初始化EventStream
    session_id = str(uuid.uuid4())
    event_stream = EventStream(sid=session_id, file_store=file_store)
    
    # 模拟代码搜索操作
    repo_path = os.getcwd()
    
    # 创建模拟的代码搜索结果
    code_search_results = [
        {
            "file": "openhands/events/action/code_search.py",
            "score": 0.95,
            "content": "class CodeSearchAction(Action):\n    \"\"\"Search for relevant code in a codebase using semantic search.\"\"\"\n    # ... code content ..."
        },
        {
            "file": "openhands/events/observation/code_search.py",
            "score": 0.92,
            "content": "class CodeSearchObservation(Observation):\n    \"\"\"Result of a code search operation.\"\"\"\n    # ... code content ..."
        }
    ]
    
    # 创建代码搜索动作
    code_search_action = CodeSearchAction(
        query="Find relevant code for RAG integration",
        repo_path=repo_path,
        extensions=[".py"],
        k=3,
        thought="I should search for relevant code to understand this task"
    )
    
    # 将动作添加到事件流
    event_stream.add_event(code_search_action, EventSource.AGENT)
    
    # 生成内容
    content = "\n".join([
        f"Result {i+1}: {result['file']} (Relevance score: {result['score']})" + 
        "\n```\n" + result['content'] + "\n```\n"
        for i, result in enumerate(code_search_results)
    ])
    
    # 创建代码搜索观察
    code_search_observation = CodeSearchObservation(
        results=code_search_results,
        content=content
    )
    
    # 将观察添加到事件流
    event_stream.add_event(code_search_observation, EventSource.ENVIRONMENT)
    
    # 打印结果
    print("\n" + "="*80)
    print("RAG代码搜索集成测试")
    print("="*80)
    print(f"代码搜索动作: {code_search_action}")
    print(f"代码搜索观察: {code_search_observation}")
    print(f"观察内容: {code_search_observation.content}")
    print("="*80)
    
    # 清理临时目录
    temp_dir.cleanup()
    
    return True

if __name__ == "__main__":
    success = test_mock_rag()
    print(f"测试{'成功' if success else '失败'}")
    sys.exit(0 if success else 1)
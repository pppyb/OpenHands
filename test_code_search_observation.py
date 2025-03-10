#!/usr/bin/env python3
"""
测试CodeSearchObservation类。
"""

import sys
from typing import Dict, List, Any

# 定义一个简单的Event类
class Event:
    pass

# 定义一个简单的ObservationType
class ObservationType:
    CODE_SEARCH = "code_search"

# 定义一个简单的Observation类
class Observation(Event):
    def __init__(self, content=""):
        self.content = content

# 定义CodeSearchObservation类
class CodeSearchObservation(Observation):
    """Result of a code search operation."""
    
    def __init__(self, results: List[Dict[str, Any]], content: str = ""):
        super().__init__(content)
        self.results = results
        self.observation = ObservationType.CODE_SEARCH
        
        if not content:
            self._generate_content()
    
    def _generate_content(self):
        """Generate formatted content from search results."""
        if not self.results:
            self.content = "No code snippets matching your query were found."
            return
        
        output = []
        for i, result in enumerate(self.results, 1):
            output.append(f"Result {i}: {result['file']} (Relevance score: {result['score']})")
            output.append("```")
            output.append(result['content'])
            output.append("```\n")
        
        self.content = "\n".join(output)
    
    @property
    def message(self) -> str:
        """Get a human-readable message describing the code search results."""
        return f'Found {len(self.results)} code snippets.'
    
    def __str__(self) -> str:
        """Get a string representation of the code search observation."""
        return f"[Found {len(self.results)} code snippets.]\n{self.content}"

# 测试函数
def test_code_search_observation():
    """测试CodeSearchObservation类。"""
    
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
    
    # 测试1：使用默认内容
    observation1 = CodeSearchObservation(results=code_search_results)
    print("\n" + "="*80)
    print("测试1：使用默认内容")
    print("="*80)
    print(f"观察内容: {observation1.content}")
    
    # 测试2：提供自定义内容
    custom_content = "这是自定义内容"
    observation2 = CodeSearchObservation(results=code_search_results, content=custom_content)
    print("\n" + "="*80)
    print("测试2：提供自定义内容")
    print("="*80)
    print(f"观察内容: {observation2.content}")
    
    return True

if __name__ == "__main__":
    success = test_code_search_observation()
    print(f"\n测试{'成功' if success else '失败'}")
    sys.exit(0 if success else 1)
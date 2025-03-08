# 文件: /workspace/OpenHands/scripts/test_code_search_integration.py

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openhands.events.action.code_search import CodeSearchAction
from openhands.runtime.action_execution_server import handle_action


def main():
    """测试代码搜索功能的集成。"""
    # 使用当前目录作为测试仓库
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 创建动作
    action = CodeSearchAction(
        query="代码搜索功能",
        repo_path=repo_path,
        extensions=['.py'],
        k=3
    )
    
    print(f"搜索内容: {action.query}")
    print(f"仓库: {action.repo_path}")
    print(f"扩展名: {', '.join(action.extensions)}")
    print("-" * 80)
    
    # 执行动作
    observation = handle_action(action)
    
    # 打印结果
    print(f"观察类型: {observation.observation}")
    print(observation)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""测试代码搜索功能的脚本。"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.events.observation.error import ErrorObservation


async def test_code_search(repo_path, query, extensions=None, k=5, min_score=0.5):
    """测试代码搜索功能。
    
    Args:
        repo_path: 要搜索的仓库路径。
        query: 搜索查询。
        extensions: 要搜索的文件扩展名列表。
        k: 返回的结果数量。
        min_score: 最小分数阈值。
    """
    # 导入 ActionExecutor
    from openhands.runtime.action_execution_server import ActionExecutor
    
    # 创建 ActionExecutor 实例
    executor = ActionExecutor(
        plugins_to_load=[],
        work_dir=repo_path,
        username="openhands",
        user_id=1000
    )
    
    # 初始化 ActionExecutor
    await executor.initialize()
    
    # 创建代码搜索动作
    action = CodeSearchAction(
        query=query,
        repo_path=repo_path,
        extensions=extensions or [".py"],
        k=k,
        min_score=min_score
    )
    
    print(f"搜索内容: {action.query}")
    print(f"仓库: {action.repo_path}")
    print(f"扩展名: {', '.join(action.extensions)}")
    print(f"最大结果数: {action.k}")
    print(f"最小分数: {action.min_score}")
    print("-" * 80)
    
    # 执行动作
    observation = await executor.code_search(action)
    
    # 打印结果
    if isinstance(observation, CodeSearchObservation):
        print(f"找到 {len(observation.results)} 个结果:")
        for i, result in enumerate(observation.results, 1):
            print(f"\n结果 {i}: {result['file']} (分数: {result['score']})")
            print("-" * 40)
            print(result['content'])
    elif isinstance(observation, ErrorObservation):
        print(f"错误: {observation.error}")
    else:
        print(f"未知观察类型: {type(observation)}")
    
    # 关闭 ActionExecutor
    executor.close()


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description='测试代码搜索功能')
    parser.add_argument('--repo', default=os.getcwd(), help='要搜索的仓库路径')
    parser.add_argument('--query', required=True, help='搜索查询')
    parser.add_argument('--extensions', nargs='+', default=['.py'], help='要搜索的文件扩展名')
    parser.add_argument('--results', type=int, default=5, help='返回的结果数量')
    parser.add_argument('--min-score', type=float, default=0.5, help='最小分数阈值')
    
    args = parser.parse_args()
    
    # 运行测试
    asyncio.run(test_code_search(
        repo_path=args.repo,
        query=args.query,
        extensions=args.extensions,
        k=args.results,
        min_score=args.min_score
    ))


if __name__ == "__main__":
    main()
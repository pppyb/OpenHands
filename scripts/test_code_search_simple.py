#!/usr/bin/env python3
"""简单测试代码搜索功能的脚本。"""

import argparse
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入 code_search_tool 函数
from openhands_aci.tools.code_search_tool import code_search_tool


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description='简单测试代码搜索功能')
    parser.add_argument('--repo', default=os.getcwd(), help='要搜索的仓库路径')
    parser.add_argument('--query', required=True, help='搜索查询')
    parser.add_argument('--extensions', nargs='+', default=['.py'], help='要搜索的文件扩展名')
    parser.add_argument('--results', type=int, default=5, help='返回的结果数量')
    parser.add_argument('--min-score', type=float, default=0.5, help='最小分数阈值')
    
    args = parser.parse_args()
    
    print(f"搜索内容: {args.query}")
    print(f"仓库: {args.repo}")
    print(f"扩展名: {', '.join(args.extensions)}")
    print(f"最大结果数: {args.results}")
    print(f"最小分数: {args.min_score}")
    print("-" * 80)
    
    # 执行代码搜索
    result = code_search_tool(
        query=args.query,
        repo_path=args.repo,
        extensions=args.extensions,
        k=args.results,
        remove_duplicates=True,
        min_score=args.min_score
    )
    
    # 打印结果
    print(f"\n搜索状态: {result['status']}")
    if result['status'] == 'success':
        print(f"找到 {len(result['results'])} 个结果:")
        for i, res in enumerate(result['results'], 1):
            print(f"\n结果 {i}: {res['file']} (分数: {res['score']})")
            print("-" * 40)
            print(res['content'])
    else:
        print(f"错误: {result.get('message', '未知错误')}")


if __name__ == "__main__":
    main()
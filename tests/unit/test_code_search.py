"""代码搜索功能的单元测试。"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from git import Repo

from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.events.observation.error import ErrorObservation


# 创建一个测试仓库的 fixture
@pytest.fixture
def test_repo():
    """创建一个包含测试文件的临时 Git 仓库。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 初始化 Git 仓库
        repo = Repo.init(temp_dir)

        # 创建测试文件
        files = {
            'main.py': 'def hello():\n    print("你好，世界!")',
            'utils/helper.py': 'def add(a, b):\n    """将两个数相加并返回结果。"""\n    return a + b',
            'auth.py': 'def authenticate(username, password):\n    """使用用户名和密码验证用户。"""\n    return username == "admin" and password == "secret"',
            'README.md': '# 测试仓库\n 这是一个测试。',
        }

        for path, content in files.items():
            file_path = Path(temp_dir) / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)

        # 添加并提交文件
        repo.index.add('*')
        repo.index.commit('初始提交')

        yield temp_dir


# 测试 CodeSearchAction 的创建
def test_code_search_action_creation():
    """测试创建 CodeSearchAction。"""
    action = CodeSearchAction(
        query="计算两个数的函数",
        repo_path="/path/to/repo",
        extensions=[".py", ".js"],
        k=5
    )
    
    assert action.query == "计算两个数的函数"
    assert action.repo_path == "/path/to/repo"
    assert action.extensions == [".py", ".js"]
    assert action.k == 5
    assert action.action == "code_search"


# 测试 CodeSearchObservation 的创建
def test_code_search_observation_creation():
    """测试创建 CodeSearchObservation。"""
    results = [
        {
            "file": "utils/helper.py",
            "score": 0.85,
            "content": 'def add(a, b):\n    """将两个数相加并返回结果。"""\n    return a + b'
        }
    ]
    
    observation = CodeSearchObservation(results=results)
    
    assert observation.results == results
    assert observation.observation == "code_search"
    assert "找到 1 个代码片段" in observation.message
    assert "utils/helper.py" in str(observation)
    assert "相关性分数: 0.85" in str(observation)


# 测试 code_search_tool 函数
@patch('openhands_aci.tools.code_search_tool.code_search_tool')
def test_code_search_tool_mock(mock_tool):
    """测试 code_search_tool 函数（使用 mock）。"""
    # 设置 mock 的返回值
    mock_tool.return_value = {
        "status": "success",
        "results": [
            {
                "file": "utils/helper.py",
                "score": 0.85,
                "content": 'def add(a, b):\n    """将两个数相加并返回结果。"""\n    return a + b'
            }
        ]
    }
    
    # 导入 code_search_tool 函数
    from openhands_aci.tools.code_search_tool import code_search_tool
    
    # 调用函数
    result = code_search_tool(
        query="计算两个数的函数",
        repo_path="/path/to/repo",
        extensions=[".py"],
        k=3
    )
    
    # 验证结果
    assert result["status"] == "success"
    assert len(result["results"]) == 1
    assert result["results"][0]["file"] == "utils/helper.py"
    assert result["results"][0]["score"] == 0.85
    
    # 验证 mock 被正确调用
    mock_tool.assert_called_once_with(
        query="计算两个数的函数",
        repo_path="/path/to/repo",
        extensions=[".py"],
        k=3,
        remove_duplicates=None,
        min_score=None
    )


# 测试实际的代码搜索功能（需要安装 sentence-transformers 和 faiss-cpu）
@pytest.mark.skipif(
    not os.environ.get("RUN_REAL_CODE_SEARCH_TEST"),
    reason="需要设置 RUN_REAL_CODE_SEARCH_TEST 环境变量才能运行实际的代码搜索测试"
)
def test_real_code_search(test_repo):
    """测试实际的代码搜索功能。"""
    # 导入 code_search_tool 函数
    from openhands_aci.tools.code_search_tool import code_search_tool
    
    # 调用函数
    result = code_search_tool(
        query="计算两个数的函数",
        repo_path=test_repo,
        extensions=[".py"],
        k=3
    )
    
    # 验证结果
    assert result["status"] == "success"
    assert len(result["results"]) > 0
    
    # 验证找到了 add 函数
    found_add = False
    for res in result["results"]:
        if "add" in res["content"]:
            found_add = True
            break
    assert found_add
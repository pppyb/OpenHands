# RAG 代码搜索功能

这个功能使用检索增强生成（Retrieval Augmented Generation，RAG）技术来实现代码搜索。它允许用户使用自然语言查询来搜索代码库中的相关代码片段。

## 功能特点

- 使用语义搜索而不是简单的关键词匹配
- 自动为代码库创建索引
- 支持多种编程语言和文件类型
- 可以过滤重复结果和低质量匹配

## 安装

此功能已集成到OpenHands中，依赖于`openhands-aci`库的`implement-rag-code-search`分支。

```bash
# 安装依赖
pip install sentence-transformers faiss-cpu
```

## 使用方法

### 作为独立脚本使用

```bash
# 运行示例脚本
python examples/rag_code_search_example.py /path/to/your/repo "你的搜索查询"
```

### 在OpenHands中使用

在OpenHands中，可以通过`code_search_tool`函数使用此功能：

```python
from openhands.runtime.plugins.agent_skills.code_search.tool import code_search_tool

# 搜索代码
result = code_search_tool(
    query="如何处理API请求",
    repo_path="/path/to/your/repo",
    k=5,  # 返回结果数量
    remove_duplicates=True,  # 移除重复文件
    min_score=0.5  # 最小相似度阈值
)

# 处理结果
if result["status"] == "success":
    print(result["formatted_output"])
else:
    print(f"搜索错误: {result['message']}")
```

## 测试

可以使用提供的测试脚本来测试功能：

```bash
python examples/test_rag_code_search.py /path/to/your/repo "如何处理API请求"
```

## 注意事项

- 首次运行时会创建索引，这可能需要一些时间
- 索引保存在代码库的`.code_search_index`目录中
- 默认支持的文件类型包括：`.py`, `.js`, `.html`, `.tsx`, `.jsx`, `.ts`, `.css`, `.md`
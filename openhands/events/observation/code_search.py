"""代码搜索观察模块。"""

from dataclasses import dataclass
from typing import Any, Dict, List

from openhands.core.schema.observation import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class CodeSearchObservation(Observation):
    """代码搜索操作的结果。
    
    该观察包含语义代码搜索操作的结果，包括文件路径、相关性分数和代码片段。
    
    属性:
        results: 搜索结果字典列表。
        observation: 观察类型。
    """
    
    results: List[Dict[str, Any]]
    observation: str = ObservationType.CODE_SEARCH
    _content: str | None = None
    
    @property
    def message(self) -> str:
        """获取描述代码搜索结果的人类可读消息。"""
        return f'找到 {len(self.results)} 个代码片段。'
    
    @property
    def content(self) -> str:
        """格式化搜索结果以供显示。"""
        if self._content is not None:
            return self._content
            
        if not self.results:
            self._content = "没有找到与您的查询匹配的代码片段。"
            return self._content
        
        output = []
        for i, result in enumerate(self.results, 1):
            output.append(f"结果 {i}: {result['file']} (相关性分数: {result['score']})")
            output.append("```")
            output.append(result['content'])
            output.append("```\n")
        
        self._content = "\n".join(output)
        return self._content
    
    def __str__(self) -> str:
        """获取代码搜索观察的字符串表示。"""
        return f"[找到 {len(self.results)} 个代码片段。]\n{self.content}"
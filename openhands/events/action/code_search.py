"""代码搜索动作模块。"""

from dataclasses import dataclass
from typing import ClassVar, List, Optional

from openhands.core.schema.action import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class CodeSearchAction(Action):
    """使用语义搜索在代码库中搜索相关代码。
    
    该动作使用检索增强生成（RAG）技术，基于自然语言查询找到相关代码。
    它首先会索引代码库（如果需要），然后执行语义搜索。
    
    属性:
        query: 自然语言查询。
        repo_path: 要搜索的Git仓库路径（如果save_dir已存在则可选）。
        save_dir: 保存/加载搜索索引的目录（默认为.code_search_index）。
        extensions: 要包含的文件扩展名列表（例如[".py", ".js"]）。
        k: 返回的结果数量。
        remove_duplicates: 是否移除重复的文件结果。
        min_score: 过滤低质量匹配的最小分数阈值。
        thought: 搜索背后的推理。
        action: 执行的动作类型。
        runnable: 指示动作是否可执行。
        security_risk: 指示与动作相关的任何安全风险。
        blocking: 指示动作是否为阻塞操作。
    """
    
    query: str
    repo_path: Optional[str] = None
    save_dir: Optional[str] = None
    extensions: Optional[List[str]] = None
    k: int = 5
    remove_duplicates: bool = True
    min_score: float = 0.5
    thought: str = ''
    action: str = ActionType.CODE_SEARCH
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    blocking: bool = True  # 设置为阻塞操作
    
    @property
    def message(self) -> str:
        """获取描述代码搜索动作的人类可读消息。"""
        return f'搜索代码: {self.query}'
    
    def __repr__(self) -> str:
        """获取代码搜索动作的字符串表示。"""
        ret = '**代码搜索动作**\n'
        ret += f'查询: {self.query}\n'
        if self.repo_path:
            ret += f'仓库: {self.repo_path}\n'
        if self.extensions:
            ret += f'扩展名: {", ".join(self.extensions)}\n'
        ret += f'返回结果数: {self.k}\n'
        ret += f'思考: {self.thought}\n'
        return ret
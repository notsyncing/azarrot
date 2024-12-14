from dataclasses import dataclass
from typing import Any

from azarrot.tools.internal import INTERNAL_TOOL_RAG_SEARCH
from azarrot.tools.tool import Tool, ToolDescription, ToolParameter


@dataclass
class FileSearchToolConfigs:
    vector_stores: list[str]
    max_result_count: int = 20
    reranker_model_id: str | None = None
    reranker_score_threshold: float = 0.0


@dataclass
class FileSearchResult:
    file_id: str
    file_name: str | None
    score: float
    matched_chunks: list[str]


@dataclass
class FileSearchOutputs:
    current_configs: FileSearchToolConfigs
    search_results: list[FileSearchResult]


class FileSearchTool(Tool):
    def description(self) -> ToolDescription:
        return ToolDescription(
            name=INTERNAL_TOOL_RAG_SEARCH,
            default_locale="zh-cn",
            display_name={"zh-cn": "文件搜索工具"},
            description={"zh-cn": "用于在一系列文件中搜索与特定句子相关的内容的工具"},
            parameters=[
                ToolParameter(
                    name="search_configs",
                    type="object",
                    description={"zh-cn": "搜索配置参数"},
                    required=True,
                    should_preset=True,
                ),
                ToolParameter(name="query", type="string", description={"zh-cn": "要搜索的句子"}, required=True),
            ],
        )

    def execute(self, **kwargs: Any) -> Any:
        return kwargs["query"]

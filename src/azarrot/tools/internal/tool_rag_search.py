from dataclasses import dataclass
from typing import Any, override

from azarrot.tools.internal import INTERNAL_TOOL_RAG_SEARCH
from azarrot.tools.tool import Tool, ToolDescription, ToolParameter


@dataclass
class RagSearchToolConfigs:
    max_result_count: int = 20
    reranker_model_id: str | None = None
    reranker_score_threshold: float = 0.0


@dataclass
class RagSearchToolResources:
    vector_stores: list[str]


@dataclass
class RagSearchResult:
    file_id: str
    file_name: str | None
    score: float
    matched_chunks: list[str]


@dataclass
class RagSearchOutputs:
    current_configs: RagSearchToolConfigs
    search_results: list[RagSearchResult]


class RagSearchTool(Tool):
    @override
    @staticmethod
    def description() -> ToolDescription:
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

    @override
    def execute(self, **kwargs: Any) -> Any:
        raise NotImplementedError

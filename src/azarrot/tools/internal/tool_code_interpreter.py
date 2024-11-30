from dataclasses import dataclass
from typing import Any

from azarrot.tools.internal import INTERNAL_TOOL_CODE_INTERPRETER
from azarrot.tools.tool import Tool, ToolDescription, ToolParameter


@dataclass
class CodeInterpreterToolConfigs:
    exposed_files: list[str] | None = None


class CodeInterpreterTool(Tool):
    def description(self) -> ToolDescription:
        return ToolDescription(
            name=INTERNAL_TOOL_CODE_INTERPRETER,
            default_locale="zh-cn",
            display_name={"zh-cn": "代码解释器"},
            description={"zh-cn": "用于解释执行Python代码的工具"},
            parameters=[
                ToolParameter(
                    name="configs",
                    type="string",
                    description={"zh-cn": "代码解释器的配置参数"},
                    required=True,
                    should_preset=True,
                ),
                ToolParameter(name="code", type="string", description={"zh-cn": "要执行的Python代码"}, required=True),
            ],
        )

    def execute(self, **kwargs: Any) -> Any:
        return kwargs["code"]

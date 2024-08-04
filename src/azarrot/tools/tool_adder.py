from typing import Any

from azarrot.tools.tool import Tool, ToolDescription, ToolParameter


class AdderTool(Tool):
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="calculator",
            default_locale="zh-cn",
            display_name={"zh-cn": "加法器"},
            description={"zh-cn": "用于将两个数字相加并返回它们的和的工具"},
            parameters=[
                ToolParameter(name="first_value", type="number", description={"zh-cn": "第一个被加数"}, required=True),
                ToolParameter(name="second_value", type="number", description={"zh-cn": "第二个被加数"}, required=True),
            ],
        )

    def execute(self, **kwargs: Any) -> Any:
        return kwargs["first_value"] + kwargs["second_value"]

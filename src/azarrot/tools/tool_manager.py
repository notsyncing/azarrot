import logging
from typing import ClassVar

from azarrot.tools.tool import Tool, ToolDescription


class ToolManager:
    _log = logging.getLogger(__name__)
    _tools: ClassVar[dict[str, Tool]] = {}

    def __init__(self) -> None:
        pass

    def register_tool(self, tool: Tool) -> None:
        desc = tool.description()

        if desc.name in self._tools:
            raise ValueError("Tool %s is already registered as %s", desc.name, self._tools[desc.name])

        self._tools[desc.name] = tool
        self._log.info("Registered tool %s", desc.name)

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tool_description(self, name: str) -> ToolDescription | None:
        tool = self.get_tool(name)

        if tool is None:
            return None

        return tool.description()

    def get_tool_list(self) -> list[Tool]:
        return list(self._tools.values())

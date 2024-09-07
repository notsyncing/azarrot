import logging
from typing import Any

from azarrot.tools.tool import Tool, ToolDescription


class ToolManager:
    _log = logging.getLogger(__name__)
    _tools: dict[str, Tool]

    def __init__(self) -> None:
        self._tools = {}

    def register_tool(self, tool: Tool) -> None:
        desc = tool.description()

        if desc.name in self._tools:
            raise ValueError("Tool %s is already registered as %s", desc.name, self._tools[desc.name])

        self._tools[desc.name] = tool
        self._log.info("Registered tool %s", desc.name)

    def unregister_tool(self, tool_name: str, fail_if_not_exists: bool = True) -> None:
        if tool_name not in self._tools:
            if fail_if_not_exists:
                raise ValueError("Tool %s is not registered", tool_name)

            return

        del self._tools[tool_name]

    def clear_registered_tools(self) -> None:
        self._tools.clear()

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tool_description(self, name: str) -> ToolDescription | None:
        tool = self.get_tool(name)

        if tool is None:
            return None

        return tool.description()

    def get_tool_list(self) -> list[Tool]:
        return list(self._tools.values())

    def is_internal_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any | None:
        tool = self.get_tool(name)

        if tool is None:
            self._log.warning("Tool %s to execute does not exist!", name)
            return None

        self._log.info("Executing tool %s", name)

        return tool.execute(**arguments)

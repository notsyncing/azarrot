from dataclasses import dataclass
from typing import Any


@dataclass
class AgentToolRequest:
    tool_name: str
    tool_preset_parameters: dict[str, Any] | None = None
    is_internal_tool: bool = False

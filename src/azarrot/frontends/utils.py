from typing import Any

from azarrot.tools.tool import LocalizedToolParameter


def to_backend_tool_parameters(tool_parameters: dict[str, Any] | None) -> list[LocalizedToolParameter]:
    if tool_parameters is None:
        return []

    param_type = tool_parameters["type"]

    if param_type != "object":
        raise ValueError(f"Unsupported tool parameter type {param_type}")

    required_params = tool_parameters.get("required", [])

    params = []

    if "properties" in tool_parameters:
        for k, v in tool_parameters["properties"].items():
            p = LocalizedToolParameter(
                name=k, description=v.get("description"), type=v.get("type"), required=k in required_params
            )

            params.append(p)

    return params

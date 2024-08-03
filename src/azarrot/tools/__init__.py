from azarrot.tools.tool_adder import AdderTool
from azarrot.tools.tool_manager import ToolManager

GLOBAL_TOOL_MANAGER = ToolManager()
GLOBAL_TOOL_MANAGER.register_tool(AdderTool())

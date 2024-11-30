from azarrot.tools.internal.tool_code_file_search import FileSearchTool
from azarrot.tools.internal.tool_code_interpreter import CodeInterpreterTool
from azarrot.tools.tool_manager import ToolManager

GLOBAL_TOOL_MANAGER = ToolManager()
GLOBAL_TOOL_MANAGER.register_tool(CodeInterpreterTool())
GLOBAL_TOOL_MANAGER.register_tool(FileSearchTool())

from azarrot.tools.internal.tool_code_interpreter import CodeInterpreterTool
from azarrot.tools.internal.tool_rag_search import RagSearchTool
from azarrot.tools.tool_manager import ToolManager

GLOBAL_TOOL_MANAGER = ToolManager()
GLOBAL_TOOL_MANAGER.register_tool(CodeInterpreterTool())
GLOBAL_TOOL_MANAGER.register_tool(RagSearchTool())

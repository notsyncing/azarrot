from typing import Literal

VectorStoreExpireBaseline = Literal["create_time", "access_time", "update_time"]
VectorStoreFileState = Literal["pending", "processing", "completed", "failed", "cancelled"]
VectorStoreFileFailedReason = Literal["system_error"]

MessageContentType = Literal["text", "image_file", "tool_outputs"]

AgentChatTaskStatus = Literal[
    "pending", "in_progress", "requires_action", "expired", "completed", "truncated", "failed", "cancelled"
]
AgentChatTaskThreadHistoryStrategy = Literal["auto", "last_messages"]
AgentChatTaskRequiredAction = Literal["tool_call_request"]
AgentChatTaskDetailType = Literal["message", "tool_call"]
AgentChatTaskDetailStatus = Literal["in_progress", "completed", "failed", "cancelled", "expired"]
AgentChatTaskDetailDataType = Literal["message", "tool_call"]

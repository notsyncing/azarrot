from typing import Literal

VectorStoreExpireBaseline = Literal["create_time", "access_time", "update_time"]
VectorStoreFileState = Literal["pending", "processing", "completed", "failed", "cancelled"]
VectorStoreFileFailedReason = Literal["system_error"]

MessageContentType = Literal["text", "image_file"]

AgentChatTaskStatus = Literal["queued", "in_progress", "requires_action", "expired", "completed", "failed", "cancelled"]

from typing import Literal

VectorStoreExpireBaseline = Literal["create_time", "access_time", "update_time"]
VectorStoreFileState = Literal["pending", "processing", "completed", "failed", "cancelled"]
VectorStoreFileFailedReason = Literal["system_error"]

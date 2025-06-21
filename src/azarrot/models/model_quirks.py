from azarrot.models.supports.default_chat_support import DEFAULT_MODEL_QUIRKS

MODEL_GENERATION_QUIRKS = {
    "qwen2": DEFAULT_MODEL_QUIRKS,
    "qwen3": DEFAULT_MODEL_QUIRKS,
    "internvl": DEFAULT_MODEL_QUIRKS.extend_with(openvino_dont_patch_model_compile=True),
}

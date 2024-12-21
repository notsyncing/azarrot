from openai import OpenAI

from azarrot.backends.pytorch_backend import BACKEND_ID_PYTORCH
from azarrot.server import Server

QWEN2_CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"


def test_qwen2_5_gptq_hello(pytorch_server: Server) -> None:
    pytorch_server.model_manager.load_huggingface_model(
        QWEN2_CHAT_MODEL, BACKEND_ID_PYTORCH, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{pytorch_server.config.host}:{pytorch_server.config.port}/v1", api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL,
        messages=[{"role": "system", "content": "你是一个乐于助人的智能助理。"}, {"role": "user", "content": "你好！"}],
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    assert result.content.find("你好！") >= 0

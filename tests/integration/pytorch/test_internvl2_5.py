import logging

from openai import OpenAI

from azarrot.backends.pytorch_backend import BACKEND_ID_PYTORCH
from azarrot.server import Server

INTERNVL2_CHAT_MODEL = "OpenGVLab/InternVL2_5-1B"

log = logging.getLogger(__name__)


def test_internvl2_5_hello(pytorch_server: Server) -> None:
    pytorch_server.model_manager.load_huggingface_model(
        INTERNVL2_CHAT_MODEL, BACKEND_ID_PYTORCH, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{pytorch_server.config.host}:{pytorch_server.config.port}/v1", api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=INTERNVL2_CHAT_MODEL,
        messages=[{"role": "system", "content": "你是一个乐于助人的智能助理。"}, {"role": "user", "content": "你好！"}],
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    log.info("Output: %s", result.content)
    assert result.content.find("你好！") >= 0


def test_internvl2_5_conversation(pytorch_server: Server) -> None:
    pytorch_server.model_manager.load_huggingface_model(
        INTERNVL2_CHAT_MODEL, BACKEND_ID_PYTORCH, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{pytorch_server.config.host}:{pytorch_server.config.port}/v1", api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=INTERNVL2_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个乐于助人的智能助理。"},
            {"role": "user", "content": "请记住：红=1，绿=2"},
            {"role": "assistant", "content": "好的，我记住了。"},
            {"role": "user", "content": "请问绿=几？"},
        ],
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    log.info("Output: %s", result.content)
    assert result.content.find("绿=2") >= 0


def test_internvl2_5_image_input(pytorch_server: Server) -> None:
    pytorch_server.model_manager.load_huggingface_model(
        INTERNVL2_CHAT_MODEL, BACKEND_ID_PYTORCH, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{pytorch_server.config.host}:{pytorch_server.config.port}/v1", api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=INTERNVL2_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个乐于助人的智能助理，并且会用中文回答问题。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这张图片里是什么？"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test-resources://internvl2_test_image_1.jpg",
                        },
                    },
                ],
            },
        ],
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    log.info("Output: %s", result.content)
    assert result.content.find("红") >= 0
    assert result.content.find("熊猫") >= 0

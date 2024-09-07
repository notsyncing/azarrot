from openai import OpenAI

from azarrot.backends.ipex_llm_backend import BACKEND_ID_IPEX_LLM
from azarrot.server import Server

INTERNVL2_CHAT_MODEL = "OpenGVLab/InternVL2-8B"


def test_internvl2_hello(ipex_llm_server: Server) -> None:
    ipex_llm_server.model_manager.load_huggingface_model(
        INTERNVL2_CHAT_MODEL, BACKEND_ID_IPEX_LLM, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{ipex_llm_server.config.host}:{ipex_llm_server.config.port}/v1", api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=INTERNVL2_CHAT_MODEL,
        messages=[{"role": "system", "content": "你是一个乐于助人的智能助理。"}, {"role": "user", "content": "你好！"}],
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    assert result.content.find("你好！") >= 0


def test_internvl2_conversation(ipex_llm_server: Server) -> None:
    ipex_llm_server.model_manager.load_huggingface_model(
        INTERNVL2_CHAT_MODEL, BACKEND_ID_IPEX_LLM, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{ipex_llm_server.config.host}:{ipex_llm_server.config.port}/v1", api_key="__TEST__"
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
    assert result.content.find("绿=2") >= 0


def test_internvl2_image_input(ipex_llm_server: Server) -> None:
    ipex_llm_server.model_manager.load_huggingface_model(
        INTERNVL2_CHAT_MODEL, BACKEND_ID_IPEX_LLM, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{ipex_llm_server.config.host}:{ipex_llm_server.config.port}/v1", api_key="__TEST__"
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
    assert result.content.find("小熊猫") >= 0

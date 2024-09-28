from openai import OpenAI

from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.server import Server

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_EMBEDDING_DIMENSION = 1024


def test_bge_m3_embedding(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        BGE_M3_MODEL, BACKEND_ID_OPENVINO, "feature-extraction", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/v1", api_key="__TEST__"
    )

    embeddings = client.embeddings.create(model=BGE_M3_MODEL, input="这是一行测试文字", encoding_format="float")

    result = embeddings.data[0]
    assert result is not None
    assert result.embedding is not None
    assert len(result.embedding) == BGE_M3_EMBEDDING_DIMENSION
    assert -0.0613 <= result.embedding[0] <= -0.0606
    assert -0.0344 <= result.embedding[511] <= -0.0336
    assert 0.0283 <= result.embedding[1023] <= 0.0290


def test_bge_m3_embedding_multiple(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        BGE_M3_MODEL, BACKEND_ID_OPENVINO, "feature-extraction", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/v1", api_key="__TEST__"
    )

    embeddings = client.embeddings.create(
        model=BGE_M3_MODEL, input=["这是一行测试文字", "今天天气怎么样？"], encoding_format="float"
    )

    result1 = embeddings.data[0]
    assert result1 is not None
    assert result1.embedding is not None
    assert len(result1.embedding) == BGE_M3_EMBEDDING_DIMENSION
    assert -0.0613 <= result1.embedding[0] <= -0.0606
    assert -0.0344 <= result1.embedding[511] <= -0.0336
    assert 0.0283 <= result1.embedding[1023] <= 0.0290

    result2 = embeddings.data[1]
    assert result2 is not None
    assert result2.embedding is not None
    assert len(result2.embedding) == BGE_M3_EMBEDDING_DIMENSION
    assert -0.0229 <= result2.embedding[0] <= -0.0227
    assert -0.0071 <= result2.embedding[511] <= -0.0068
    assert -0.0303 <= result2.embedding[1023] <= -0.0301

import logging

import cohere
import requests

from azarrot.backends.sentences_transformers_backend import BACKEND_ID_SENTENCE_TRANSFORMERS
from azarrot.server import Server

BGE_RERANKER_M3_MODEL = "BAAI/bge-reranker-v2-m3"

log = logging.getLogger(__name__)


def test_bge_reranker_m3_jina_reranking(sentence_transformers_server: Server) -> None:
    sentence_transformers_server.model_manager.load_huggingface_model(
        BGE_RERANKER_M3_MODEL, BACKEND_ID_SENTENCE_TRANSFORMERS, "text-classification", skip_if_loaded=True
    )

    url = f"http://{sentence_transformers_server.config.host}:{sentence_transformers_server.config.port}/jina/v1/rerank"

    headers = {
        "Content-Type": "application/json",
    }

    documents = [
        "今天的天气好，多云转晴",
        "今天可能要下雨",
        "明天天气不太好",
        "昨天我们去了成都玩，那边出太阳了",
        "今天中午吃什么？"
    ]

    data = {
        "model": BGE_RERANKER_M3_MODEL,
        "query": "今天的天气",
        "documents": documents
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    r = response.json()

    log.info("Output: %s", response.text)
    assert r["model"] == BGE_RERANKER_M3_MODEL
    assert r["usage"]["total_tokens"] > 0
    assert len(r["results"]) == 5

    assert r["results"][0]["index"] == 0
    assert r["results"][0]["relevance_score"] > 0.7
    assert r["results"][0]["document"]["text"] == documents[0]

    assert r["results"][1]["index"] == 1
    assert r["results"][1]["relevance_score"] > 0.7
    assert r["results"][1]["document"]["text"] == documents[1]

    assert r["results"][2]["index"] == 2
    assert r["results"][2]["relevance_score"] < 0.1
    assert r["results"][2]["document"]["text"] == documents[2]

    assert r["results"][3]["index"] == 3
    assert r["results"][3]["relevance_score"] < 0.1
    assert r["results"][3]["document"]["text"] == documents[3]

    assert r["results"][4]["index"] == 4
    assert r["results"][4]["relevance_score"] < 0.1
    assert r["results"][4]["document"]["text"] == documents[4]


def test_bge_reranker_m3_cohere_reranking(sentence_transformers_server: Server) -> None:
    sentence_transformers_server.model_manager.load_huggingface_model(
        BGE_RERANKER_M3_MODEL, BACKEND_ID_SENTENCE_TRANSFORMERS, "text-classification", skip_if_loaded=True
    )

    url = f"http://{sentence_transformers_server.config.host}:{sentence_transformers_server.config.port}/cohere/"

    co = cohere.ClientV2(base_url=url, api_key="FREE")  # type: ignore[reportCallIssue]

    documents = [
        "今天的天气好，多云转晴",
        "今天可能要下雨",
        "明天天气不太好",
        "昨天我们去了成都玩，那边出太阳了",
        "今天中午吃什么？"
    ]

    resp = co.rerank(
        query="今天的天气",
        model=BGE_RERANKER_M3_MODEL,
        documents=documents,
        return_documents=True
    )

    log.info("Output: %s", resp)
    assert resp.id is not None
    assert resp.meta is not None
    assert resp.meta.billed_units is not None
    assert resp.meta.billed_units.search_units is not None
    assert resp.meta.billed_units.search_units > 0

    assert len(resp.results) == 5

    assert resp.results[0].index == 0
    assert resp.results[0].relevance_score > 0.7
    assert resp.results[0].document is not None
    assert resp.results[0].document.text == documents[0]

    assert resp.results[1].index == 1
    assert resp.results[1].relevance_score > 0.7
    assert resp.results[1].document is not None
    assert resp.results[1].document.text == documents[1]

    assert resp.results[2].index == 2
    assert resp.results[2].relevance_score < 0.1
    assert resp.results[2].document is not None
    assert resp.results[2].document.text == documents[2]

    assert resp.results[3].index == 3
    assert resp.results[3].relevance_score < 0.1
    assert resp.results[3].document is not None
    assert resp.results[3].document.text == documents[3]

    assert resp.results[4].index == 4
    assert resp.results[4].relevance_score < 0.1
    assert resp.results[4].document is not None
    assert resp.results[4].document.text == documents[4]

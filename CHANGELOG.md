# Changelog

The following are change log of each versions.

## 0.4.0 (2025-07-??)

- Basic support for OpenAI Responses API
- Support tool call streaming
- Removed implicit internal tool calling mechanism
- Support Qwen3 reasoning
- Support loading AWQ model with OpenVINO
- Removed IPEX-LLM backend
- Update OpenVINO to 2025.2.0
- Support 🤗Transformers assisted generation.
- [BREAKING] All OpenAI endpoints are prefixed with `/openai/`
- Support Jina and Cohere Rerank API
- Add sentence-transformers backend
- Update transformers to 4.52.4
- Add PyTorch backend
- [WIP] Support assistant API
- Support loading GPTQ model with `gptqmodel`
- Support vector store API
- Support uploads API
- Support files API

## 0.3.0 (2024-09-08)

- Add docker image build
- Support list input of text on embeddings API
- Support downloading model from huggingface
- Support auto-batching on chat API
- Support `top_p`, `temperature` and `seed` parameters in chat API
- Update OpenVINO to 2024.3.0
- Update IPEX-LLM to 2.1.0

## 0.2.0 (2024-08-04)

- Add IPEX-LLM backend
- Support InternVL2 on IPEX-LLM backend with OpenAI chat completion image input
- Support Qwen2 tool calling on IPEX-LLM and OpenVINO backend with OpenAI chat completion tools input
- Support embedding models on IPEX-LLM and OpenVINO backend with OpenAI embedding API
- Support parallel completion requests: concurrent completion requests can be submit on both OpenVINO and IPEX-LLM backends (not batching)
- Add README and changelog

## 0.1.0 (2024-06-30)

Initial release with OpenVINO support and basic OpenAI chat completion API.
# Azarrot

(WIP) An OpenAI compatible LLM inference server, focusing on OpenVINO™ usage.

The name `azarrot` is combined from `azalea` and `parrot`.

## Changelog

See [CHANGELOG](./CHANGELOG.md) for more details.

## Supported OpenAI features

- ✅：Fully supported
- ☑️：Mostly supported
- ❓：Implemented, but not tested, may work or not
- 🚧：Working in progress, not working yet
- ❌：Not supported yet

### Backend-specific features

|Feature|Subfeature|OpenVINO|PyTorch|SentenceTransformers|Remarks|
|-------|----------|--------|--------|--------------------|-------|
|Chat|Basic chat completion|☑️|☑️|❌|Text generation works, some advanced parameters (like `frequency_penalty`, `n`, `logprobs`, etc) not implemented yet|
|Chat|Seeding|✅|✅|❌||
|Chat|Streaming response|✅|✅|❌||
|Chat|Image input|❌|✅|❌|InternVL3 supported|
|Embeddings|Create embeddings|☑️|❌|✅|`encoding_format` not implemented yet|
|Responses|Basic chat completion|☑️|☑️|❌|Text generation works, some advanced parameters (like `top_logprobs`, etc) not implemented yet|

### Backend-agnostic features

|Feature|Subfeature|Supported|Remarks|
|-------|----------|---------|-------|
|Models|List models|✅||
|Chat|Tool calling|✅|`<tool_call>` flavors supported|
|Responses|Function calling|✅|`<tool_call>` flavors supported|
|Files|Upload, list, retrieve, delete, retrieve content|✅||
|Uploads|Create, upload, complete, cancel|✅||
|Assistants (v2)|Assistants|☑️|`response_format` not implemented yet|
|Assistants (v2)|Threads|✅||
|Assistants (v2)|Messages|✅||
|Assistants (v2)|Runs & Run steps|🚧|`include[]`, `response_format` not implemented yet, tools not implemented yet, `stream` not implemented yet, some other small parts may also not implemented|
|Assistants (v2)|Vector stores|✅|Vector store bytes used is estimated|
|Assistants (v2)|Vector store files|✅|Vector store bytes used is estimated|
|Assistants (v2)|Vector store file batches|✅|Vector store bytes used is estimated|

### Other features

- Prefix KV caching based on radix tree with LRU evicting
- Auto-batching on OpenAI Chat API
- Auto model downloading from huggingface
- Jina and Cohere(v2) Rerank API support on SentenceTransformers backend
- Assisted generation on 🤗Transformers-based backends (OpenVINO, PyTorch)

## Tested models

|Model|Repository|Device|Backend|Remarks|
|-----|----------|------|-------|-------|
|CodeQwen1.5-7B|https://huggingface.co/Qwen/CodeQwen1.5-7B|Intel GPU|PyTorch, OpenVINO||
|InternVL3-8B-Instruct|https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct|Intel GPU|PyTorch|Image input supported|
|bge-m3|https://huggingface.co/BAAI/bge-m3|Intel GPU, CPU|OpenVINO, SentenceTransformers|Accuracy may decrease if quantized to int8|
|Qwen2-7B-Instruct|https://huggingface.co/Qwen/Qwen2-7B-Instruct|Intel GPU|PyTorch|Tool calling supported|
|Qwen2.5-Coder-7B-Instruct|https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct|Intel GPU|PyTorch||
|Qwen3-3B|https://huggingface.co/Qwen/Qwen3-1.7B|Intel GPU, CPU|OpenVINO|Tool calling supported|

Other untested models may work or not.

## Prerequisites

### Hardware

Azarrot supports CPUs and Intel GPUs.

Tested GPUs:

- Intel A770 16GB
- Intel Xe 96EU (i7 12700H) (stable under OpenVINO)

### Software

We use `Python 3.12`.

Azarrot is tested on Fedora 42 and python 3.12.

## Usage

> WARNING: This project is still in early stages. Bugs are expected.

### With Docker or podman

Image: `ghcr.io/notsyncing/azarrot:main`

See `docker/docker-compose.yml` for configuration example.

### Install from PyPI

First, install azarrot from PyPI:

```bash
pip install azarrot
```

Then, create a `server.yml` in the directory you want to run it:

```bash
mkdir azarrot

# Copy from examples/server.yml
cp <SOURCE_ROOT>/examples/server.yml azarrot/
```

`<SOURCE_ROOT>` means the repository path you cloned.

In `server.yml` you can configure things like listening port, model path, etc.

Next we create the models directory:

```bash
cd azarrot
mkdir models
```

And copy an example model file into the models directory:

```bash
cp <SOURCE_ROOT>/examples/Qwen2.5-Coder-1.5B-Instruct.model.yml models/
```

Azarrot will load all `.model.yml` files in this directory.

Now we can start the server:

```bash
python -m azarrot
```

And access `http://localhost:8080/v1/models` too see all loaded models.

More details are in the documents: [Azarrot documents](https://notsyncing.github.io/azarrot/)
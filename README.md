# Azarrot

(Early WIP) An OpenAI compatible LLM inference server, focusing on OpenVINO™ and IPEX-LLM usage.

The name `azarrot` is combined from `azalea` and `parrot`.

## Motivation

NVIDIA sucks on Linux, and AMD does not like people running ROCm on their consumer cards (sadly my RX 5500 XT is not supported).
Meanwhile, Intel consumer cards are cheap, and have good fundamental software support, Intel is also actively maintaining and upstreaming many AI libraries.

So I bought an A770, but all the existing inference servers are lacking on Intel cards: some lacks quantization, some only support a few models, some does not run at all... and they are all lacking on OpenAI API features.

Finally, I decided to create my own inference server, focusing on Intel cards, and targeting full OpenAI API features.
Let's see how far could I go.

## Changelog

See [CHANGELOG](./CHANGELOG.md) for more details.

## Supported OpenAI features

- ✅：Fully supported
- ☑️：Partially supported
- ❓：Implemented, but not tested, may work or not
- 🚧：Working in progress
- ❌：Not supported yet

|Feature|Subfeature|IPEX-LLM|OpenVINO|Remarks|
|-------|----------|--------|--------|-------|
|Chat|Basic chat completion|☑️|☑️|Text generation works, some parameters (like `frequency_penalty`) not implemented yet|
|Chat|Seeding|✅|✅||
|Chat|Streaming response|✅|✅||
|Chat|Image input|✅|❌|InternVL2 supported|
|Chat|Tool calling|✅|❌|Qwen2 supported|
|Embeddings|Create embeddings|❌|☑️|`encoding_format` not implemented yet|
|Models|List models|✅|✅||

### Other features

- Internal tool calling on OpenAI Chat API without explicit tool calling request
- Auto-batching on OpenAI Chat API
- Auto model downloading from huggingface

## Tested models

|Model|Repository|Device|Backend|Remarks|
|-----|----------|------|-------|-------|
|CodeQwen1.5-7B|https://huggingface.co/Qwen/CodeQwen1.5-7B|Intel GPU|IPEX-LLM, OpenVINO||
|InternVL2-8B|https://huggingface.co/OpenGVLab/InternVL2-8B|Intel GPU|IPEX-LLM|Image input supported|
|bge-m3|https://huggingface.co/BAAI/bge-m3|Intel GPU, CPU|OpenVINO|Accuracy may decrease if quantized to int8|
|Qwen2-7B-Instruct|https://huggingface.co/Qwen/Qwen2-7B-Instruct|Intel GPU|IPEX-LLM|Tool calling supported|

Other untested models may work or not.

## Prerequisites

### Hardware

Azarrot supports CPUs and Intel GPUs. NVIDIA and AMD GPUs may work if you manually install corresponding `torch` libraries.

Tested GPUs:

- Intel A770 16GB
- Intel Xe 96EU (i7 12700H)

### Software

Due to the `xpu` branch of `intel-extension-for-pytorch` still has no python 3.12 build, we have to use `Python 3.11` or below.

You also have to install oneAPI Toolkit (at least 2024.0) and drivers.

Azarrot is tested on Ubuntu 22.04 and python 3.10.

## Usage

> WARNING: This project is still in early stages. Bugs are expected.

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
cp <SOURCE_ROOT>/examples/CodeQwen1.5-7B-ipex-llm.model.yml models/
```

Azarrot will load all `.model.yml` files in this directory.
You need to manually download the model from huggingface, or convert them if you are using the OpenVINO backend:

```bash
huggingface-cli download --local-dir models/CodeQwen1.5-7B Qwen/CodeQwen1.5-7B
```

Azarrot will convert it to `int4` when loading the model.

Now we can start the server:

```bash
source /opt/intel/oneapi/setvars.sh
python -m azarrot
```

And access `http://localhost:8080/v1/models` too see all loaded models.

More details are in the documents: [Azarrot documents](https://notsyncing.github.io/azarrot/)
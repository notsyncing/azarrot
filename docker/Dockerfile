FROM docker.io/intel/oneapi-basekit:2024.2.1-0-devel-ubuntu22.04

RUN apt update && apt -y upgrade && apt -y install python3 python3-pip

RUN mkdir /app && mkdir /app/src && mkdir /app/models && mkdir /app/working
WORKDIR /app/working

COPY src /app/src/
COPY pyproject.toml README.md LICENSE docker/server.yml /app/

# Change the URL to https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ if you are in China
ENV PIP_EXTRA_INDEX_URL https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# We need to override some too-old dependencies on ipex-llm
RUN cd /app && python3 -m pip install -U pip && python3 -m pip install . && \
    python3 -m pip install torch==2.1.0.post3 torchvision==0.16.0.post3 torchaudio==2.1.0.post3 intel-extension-for-pytorch==2.1.40+xpu oneccl_bind_pt==2.1.400+xpu && \
    python3 -m pip install numpy==1.26.4 && \
    python3 -m pip install transformers==4.41.2 trl

VOLUME /app/models
VOLUME /app/working
EXPOSE 8080

# Workaround for https://github.com/intel/compute-runtime/issues/710
ENV NEOReadDebugKeys 1
ENV OverrideGpuAddressSpace 48

CMD python3 -m azarrot --config /app/server.yml

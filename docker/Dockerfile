FROM docker.io/intel/oneapi-basekit:2025.0.1-0-devel-ubuntu22.04

RUN apt update && apt -y install python3 python3-pip libmagic1

RUN mkdir /app && mkdir /app/src && mkdir /app/models && mkdir /app/working
WORKDIR /app/working

COPY src /app/src/
COPY pyproject.toml README.md LICENSE docker/server.yml /app/

# Change the URL to https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ if you are not in mainland China
ENV PIP_EXTRA_INDEX_URL https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

# We need to override some too-old dependencies on ipex-llm
RUN cd /app && python3 -m pip install -U pip && python3 -m pip install . && python3 -m pip install .[model_supports] && \
    python3 -m pip install auto-gptq==0.7.1 && \
    python3 -m pip cache purge && \
    python3 -m pip install transformers==4.46.3 trl sentence-transformers==3.3.1

VOLUME /app/models
VOLUME /app/working
EXPOSE 8080

# Workaround for https://github.com/intel/compute-runtime/issues/710
ENV NEOReadDebugKeys 1
ENV OverrideGpuAddressSpace 48

CMD python3 -m azarrot --config /app/server.yml

Quickstart
==========

This page will guide you to start using azarrot.

Prerequisites
-------------

Azarrot has some prerequisites for your hardware and software.

Hardware
^^^^^^^^

Azarrot supports CPUs and Intel GPUs.

Tested GPUs:

* Intel A770 16GB
* Intel Xe 96EU (i7 12700H)

Other devices should work if they are supported by oneAPI toolkit and drivers.

Software
^^^^^^^^

* Any Linux distribution
* Intel GPU drivers (if you are using Intel GPUs) from https://dgpu-docs.intel.com/driver/client/overview.html
* Intel oneAPI Base Toolkit 2024.0 or above from https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/
* Python 3.11.x or below

Azarrot is developed and tested on Ubuntu 22.04 and python 3.10.

Install
-------

With Docker or podman
^^^^^^^^^^^^^^^^^^^^^

Image: `ghcr.io/notsyncing/azarrot:latest`

See `docker/docker-compose.yml` for example.

Install from PyPI
^^^^^^^^^^^^^^^^^

Simply install azarrot from PyPI:

.. code-block:: bash

    pip install azarrot

Then, create a `server.yml` in the directory you want to run it:

.. code-block:: bash

    mkdir azarrot

    # Copy from examples/server.yml
    cp <SOURCE_ROOT>/examples/server.yml azarrot/

`<SOURCE_ROOT>` means the repository path you cloned.

In `server.yml` you can configure things like listening port, model path, etc.

Next we create the models directory:

.. code-block:: bash

    cd azarrot
    mkdir models

And copy an example model file into the models directory:

.. code-block:: bash

    cp <SOURCE_ROOT>/examples/CodeQwen1.5-7B-ipex-llm.model.yml models/

Azarrot will load all `.model.yml` files in this directory.
You need to manually download the model from huggingface, or convert them if you are using the OpenVINO backend:

.. code-block:: bash

    huggingface-cli download --local-dir models/CodeQwen1.5-7B Qwen/CodeQwen1.5-7B

Azarrot will convert it to `int4` when loading the model with IPEX-LLM backend.

Start to use
------------

Now we can start the server:

.. code-block:: bash

    source /opt/intel/oneapi/setvars.sh
    python -m azarrot

And access `http://localhost:8080/v1/models` too see all loaded models.
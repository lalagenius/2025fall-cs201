# Build LLM在本地 Mac 运行

Updated 15:23 GMT+8 Aug 12, 2025

2025 summer, Complied by Hongfei Yan



《从零构建大模型》代码，https://github.com/rasbt/LLMs-from-scratch

Build a Large Language Model (From Scratch)



# Setup

#Install Python

#python --version

brew install python@3.11

> Python is installed as
>  /opt/homebrew/bin/python3.11

pip3.11 install uv



从 https://github.com/rasbt/LLMs-from-scratch 下载 LLMs-from-scratch-main.zip，移到Desktop目录并解开。

创建虚拟环境

cd ~/Desktop/LLMs-from-scratch-main

uv venv --python=python3.11

> Using CPython 3.11.13 interpreter at: /opt/homebrew/opt/python@3.11/bin/python3.11
> Creating virtual environment at: .venv
> Activate with: source .venv/bin/activate

Activate with: source .venv/bin/activate

which python

> /Users/hfyan/Desktop/LLMs-from-scratch-main/.venv/bin/python

python --version

> Python 3.11.13

uv pip install packages

> Resolved 13 packages in 3.42s
> Prepared 13 packages in 9.15s
> Installed 13 packages in 18ms
>  \+ cached-property==1.5.2
>  \+ certifi==2025.7.14
>  \+ charset-normalizer==3.4.2
>  \+ click==7.1.2
>  \+ idna==3.10
>  \+ lxml==4.9.4
>  \+ packages==0.1.1
>  \+ pymongo==3.13.0
>  \+ redis==3.5.3
>  \+ requests==2.32.4
>  \+ requests-cache==0.5.2
>  \+ tqdm==4.67.1
>  \+ urllib3==2.5.0

uv pip install torch

> Resolved 9 packages in 951ms
> Prepared 9 packages in 54.58s
> Installed 9 packages in 231ms
>  \+ filelock==3.18.0
>  \+ fsspec==2025.7.0
>  \+ jinja2==3.1.6
>  \+ markupsafe==3.0.2
>  \+ mpmath==1.3.0
>  \+ networkx==3.5
>  \+ sympy==1.14.0
>  \+ torch==2.7.1
>  \+ typing-extensions==4.14.1

uv pip install -r requirements.txt 

> Resolved 140 packages in 5.26s
> Prepared 125 packages in 5m 28s
> Installed 125 packages in 543ms
>  \+ absl-py==2.3.1
>  \+ anyio==4.9.0
>  \+ appnope==0.1.4
>  \+ argon2-cffi==25.1.0
>  \+ argon2-cffi-bindings==21.2.0
>  \+ arrow==1.3.0
>  \+ asttokens==3.0.0
>  \+ astunparse==1.6.3
>  \+ async-lru==2.0.5
>  \+ attrs==25.3.0
>  \+ babel==2.17.0
>  \+ beautifulsoup4==4.13.4
>  \+ bleach==6.2.0
>  \+ cffi==1.17.1
>  \+ comm==0.2.2
>  \+ contourpy==1.3.2
>  \+ cycler==0.12.1
>  \+ debugpy==1.8.15
>  \+ decorator==5.2.1
>  \+ defusedxml==0.7.1
>  \+ executing==2.2.0
>  \+ fastjsonschema==2.21.1
>  \+ flatbuffers==25.2.10
>  \+ fonttools==4.59.0
>  \+ fqdn==1.5.1
>  \+ gast==0.6.0
>  \+ google-pasta==0.2.0
>  \+ grpcio==1.73.1
>  \+ h11==0.16.0
>  \+ h5py==3.14.0
>  \+ httpcore==1.0.9
>  \+ httpx==0.28.1
>  \+ ipykernel==6.29.5
>  \+ ipython==9.4.0
>  \+ ipython-pygments-lexers==1.1.1
>  \+ isoduration==20.11.0
>  \+ jedi==0.19.2
>  \+ json5==0.12.0
>  \+ jsonpointer==3.0.0
>  \+ jsonschema==4.25.0
>  \+ jsonschema-specifications==2025.4.1
>  \+ jupyter-client==8.6.3
>  \+ jupyter-core==5.8.1
>  \+ jupyter-events==0.12.0
>  \+ jupyter-lsp==2.2.6
>  \+ jupyter-server==2.16.0
>  \+ jupyter-server-terminals==0.5.3
>  \+ jupyterlab==4.4.5
>  \+ jupyterlab-pygments==0.3.0
>  \+ jupyterlab-server==2.27.3
>  \+ keras==3.10.0
>  \+ kiwisolver==1.4.8
>  \+ lark==1.2.2
>  \+ libclang==18.1.1
>  \+ markdown==3.8.2
>  \+ markdown-it-py==3.0.0
>  \+ matplotlib==3.10.3
>  \+ matplotlib-inline==0.1.7
>  \+ mdurl==0.1.2
>  \+ mistune==3.1.3
>  \+ ml-dtypes==0.5.1
>  \+ namex==0.1.0
>  \+ nbclient==0.10.2
>  \+ nbconvert==7.16.6
>  \+ nbformat==5.10.4
>  \+ nest-asyncio==1.6.0
>  \+ notebook-shim==0.2.4
>  \+ numpy==2.0.2
>  \+ opt-einsum==3.4.0
>  \+ optree==0.16.0
>  \+ overrides==7.7.0
>  \+ packaging==25.0
>  \+ pandas==2.3.1
>  \+ pandocfilters==1.5.1
>  \+ parso==0.8.4
>  \+ pexpect==4.9.0
>  \+ pillow==11.3.0
>  \+ platformdirs==4.3.8
>  \+ prometheus-client==0.22.1
>  \+ prompt-toolkit==3.0.51
>  \+ protobuf==5.29.5
>  \+ psutil==7.0.0
>  \+ ptyprocess==0.7.0
>  \+ pure-eval==0.2.3
>  \+ pycparser==2.22
>  \+ pygments==2.19.2
>  \+ pyparsing==3.2.3
>  \+ python-dateutil==2.9.0.post0
>  \+ python-json-logger==3.3.0
>  \+ pytz==2025.2
>  \+ pyyaml==6.0.2
>  \+ pyzmq==27.0.0
>  \+ referencing==0.36.2
>  \+ regex==2024.11.6
>  \+ rfc3339-validator==0.1.4
>  \+ rfc3986-validator==0.1.1
>  \+ rfc3987-syntax==1.1.0
>  \+ rich==14.0.0
>  \+ rpds-py==0.26.0
>  \+ send2trash==1.8.3
>  \+ setuptools==80.9.0
>  \+ six==1.17.0
>  \+ sniffio==1.3.1
>  \+ soupsieve==2.7
>  \+ stack-data==0.6.3
>  \+ tensorboard==2.19.0
>  \+ tensorboard-data-server==0.7.2
>  \+ tensorflow==2.19.0
>  \+ tensorflow-io-gcs-filesystem==0.37.1
>  \+ termcolor==3.1.0
>  \+ terminado==0.18.1
>  \+ tiktoken==0.9.0
>  \+ tinycss2==1.4.0
>  \+ tornado==6.5.1
>  \+ traitlets==5.14.3
>  \+ types-python-dateutil==2.9.0.20250708
>  \+ tzdata==2025.2
>  \+ uri-template==1.3.0
>  \+ wcwidth==0.2.13
>  \+ webcolors==24.11.1
>  \+ webencodings==0.5.1
>  \+ websocket-client==1.8.0
>  \+ werkzeug==3.1.3
>  \+ wheel==0.45.1
>  \+ wrapt==1.17.2

uv run jupyter lab










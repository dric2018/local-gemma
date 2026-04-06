# local-gemma
Trying out Gemma models


## Setup
### Local Machine
* Hardware: Macbook Pro M4 (24 GB)
* Server: 
    - Llama.cpp

### Steps
Install or upgrade llama.cpp

```bash
$ brew upgrade llama.cpp 
# or brew install llama.cpp if it is not yet available on your machine
```
Launch llama.cpp

```bash
$ llama-server -hf ggml-org/gemma-4-E4B-it-GGUF:Q8_0
```

### Using docker
- Create a virtual environment

Using `Miniconda`

```bash
$ conda create -n gemma4 python=3.14 -y
# then activate it using
$ conda activate gemma4
```

Using `venv`
```bash
$ python -m venv .venv
# then activate it using
$ source .venv/bin/activate
```

Then you can install the dependencies by running:
```bash
$ pip install -r requirements.txt
# or using uv pip install -r requirements.txt
```

Launch the llama.cpp server
```bash
$ docker compose up # -d for detached launch
```

Once the server is up and running, you can interact with it through the exposed API as follows:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="local-model"
)

response = llm.invoke("Explain Docker in one sentence.")
print(response.content)
```

# Notes
- Version `ggml-org/gemma-4-E4B-it-GGUF` seem overthinking almost all questions
    - Getting an average of `11 t/s` with the specified setup
    - capabilities on African languages needs further testing, but first trials on `Kinyarwanda` was not promising
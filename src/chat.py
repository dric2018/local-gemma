from config import CFG
from langchain_openai import ChatOpenAI

# LangChain's wrapper for OpenAI-compatible APIs
llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="local-model",
    max_completion_tokens=CFG.MAX_NEW_TOKENS
)

response = llm.invoke("Explain Docker in one sentence.")
print(response.content)

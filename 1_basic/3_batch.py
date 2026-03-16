"""
使用ChatOpenRouter的batch方法批量调用模型
"""
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr
import os
from dotenv import load_dotenv

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
openrouter_model_name = os.getenv("OPENROUTER_MODEL_NAME")
if not openrouter_model_name:
    raise ValueError("OPENROUTER_MODEL_NAME is not set in the environment variables.")

llm = ChatOpenRouter(
    model=openrouter_model_name,
    api_key=SecretStr(openrouter_api_key),
    reasoning={"effort": "none", "summary": "auto"}
)

responses = llm.batch([
    "请介绍一下你自己。",
    "今天天气怎么样？",
])

for response in responses:
    print(response.content)
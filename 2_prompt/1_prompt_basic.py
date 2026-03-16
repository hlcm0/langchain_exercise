"""
使用ChatPromptTemplate构建prompt，并创建一个简单的ICEL管道
"""
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
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

prompt = ChatPromptTemplate.from_template(
    "把下面的文本翻译成英文：{input}"
)

chain = prompt | llm

response = chain.invoke({"input": "今天天气真好！"})
print(response.content)
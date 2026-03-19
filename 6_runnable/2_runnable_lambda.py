"""
RunnableLambda 示例。
把普通 Python 函数接入 Runnable 链中，先做输入清洗，再交给模型处理。
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

openrouter_model_name = os.getenv("OPENROUTER_MODEL_NAME")
if not openrouter_model_name:
    raise ValueError("OPENROUTER_MODEL_NAME is not set in the environment variables.")


def normalize_input(payload: dict) -> dict:
    cleaned_text = payload["input"].strip().replace("\n", " ")
    return {"input": cleaned_text}


llm = ChatOpenRouter(
    model=openrouter_model_name,
    api_key=SecretStr(openrouter_api_key),
    reasoning={"effort": "none", "summary": "auto"},
)

prompt = ChatPromptTemplate.from_template(
    "请把下面的文本翻译成英文，并保留原意：{input}"
)

chain = RunnableLambda(normalize_input) | prompt | llm | StrOutputParser()

response = chain.invoke({"input": "\n  逸一时，误一世。  \n"})

print("清洗并翻译后的结果：")
print(response)

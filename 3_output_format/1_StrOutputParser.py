"""
使用StrOutputParser将LLM的输出直接解析为字符串格式，适用于需要纯文本输出的场景。
"""
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"input": "今天天气真好！"})
print(response)
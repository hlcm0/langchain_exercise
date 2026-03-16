"""
使用JsonOutputParser将LLM的输出直接解析为JSON格式，适用于需要结构化数据输出的场景
不使用Provider原生的JSON解析功能
"""
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
    "将下面的用户信息解析成JSON格式，包含name, age, city三个字段：{input}"
)

chain = prompt | llm | JsonOutputParser()

response = chain.invoke({"input": "用户信息：张三，25岁，住在北京。"})
print(response)
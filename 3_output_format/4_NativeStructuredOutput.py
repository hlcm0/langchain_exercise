"""
直接使用模型的原生能力来生成结构化输出，而不是让模型生成文本后再解析。
相比使用 PydanticOutputParser 或 JsonOutputParser，
这种方式无需在prompt内插入格式说明，且模型直接原生生成结构化数据，省token，稳健型更好。
"""
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, SecretStr
import os
from dotenv import load_dotenv

class UserInfo(BaseModel):
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄")
    city: str = Field(description="用户所在城市")

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

structured_llm = llm.with_structured_output(UserInfo)

prompt = ChatPromptTemplate.from_template(
    """
    请解析用户信息为json。
    用户输入：{input}
    """
)

chain = prompt | structured_llm

response = chain.invoke(
    {
        "input": "用户信息：张三，25岁，住在北京。",
    }
)

print("Pydantic 对象:")
print(response)
"""
相比直接使用 JsonOutputParser
使用 PydanticOutputParser 可以将 LLM 输出约束为指定 schema。
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

parser = PydanticOutputParser(pydantic_object=UserInfo)

prompt = ChatPromptTemplate.from_template(
    """
    请将用户信息提取为指定 JSON。

    {format_instructions}

    用户输入：{input}
    """
)

chain = prompt | llm | parser

response = chain.invoke(
    {
        "input": "用户信息：张三，25岁，住在北京。",
        "format_instructions": parser.get_format_instructions(),
    }
)

print("Pydantic 对象:")
print(response)
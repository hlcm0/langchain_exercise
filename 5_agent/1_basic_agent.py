"""
基础的Agent示例，使用create_agent方法创建Agent，完成天气查询和结构化输出的功能

备注：
试验发现qwen3.5-plus模型受模型能力限制，无法在启用工具调用的同时使用原生的结构化输出功能
而grok模型则可以同时使用工具调用和原生的结构化输出。

对于qwen3.5-plus模型，可以分为两个阶段：第一阶段使用工具获取天气，第二阶段使用一个单独的Agent进行结构化输出，
也可以使用ToolStrategy的方式通过工具调用实现结构化输出，本代码就用的是这个方式
"""
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_openrouter import ChatOpenRouter
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from pydantic import BaseModel, Field, SecretStr
import os
from dotenv import load_dotenv

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
openrouter_model_name = os.getenv("OPENROUTER_MODEL_NAME") 
# openrouter_model_name = "x-ai/grok-4.20-beta"  # grok模型可以同时使用工具调用和原生的结构化输出功能
# openrouter_model_name = "qwen3.5-plus"  # qwen3.5-plus模型受能力限制，无法在启用工具调用的同时使用原生的结构化输出功能
if not openrouter_model_name:
    raise ValueError("OPENROUTER_MODEL_NAME is not set in the environment variables.")

llm = ChatOpenRouter(
    model=openrouter_model_name,
    api_key=SecretStr(openrouter_api_key),
    reasoning={"effort": "none", "summary": "auto"}
)

class WeatherInfo(BaseModel):
    city: str = Field(description="城市名称")
    temperature: float = Field(description="城市的温度")
    condition: str = Field(description="天气状况")

class WeatherResponse(BaseModel):
    weather: list[WeatherInfo] = Field(description="获取到的天气信息列表")

@tool
def get_weather(location: str) -> str:
    """
    获取天气信息的函数
    Args:
        location (str): 城市名称
    """
    if location == "北京":
        return WeatherInfo(city="北京", temperature=30.0, condition="晴").model_dump_json()
    elif location == "上海":
        return WeatherInfo(city="上海", temperature=28.0, condition="多云").model_dump_json()
    else:
        return WeatherInfo(city=location, temperature=25.0, condition="未知").model_dump_json()

# 能力强的模型这里可以用ProviderStrategy的方式让模型直接输出原生的结构化结果，但qwen不太行
agent = create_agent(model=llm, tools=[get_weather], response_format=ToolStrategy(WeatherResponse))

conversation = [
    SystemMessage(content="你是一个天气预报助手，可以调用工具获取天气信息，并以JSON格式返回。"),
    HumanMessage(content="北京和上海什么天气？")
]

response = agent.invoke({'messages': conversation})

print("Agent结构化结果：")
print(response['structured_response'])
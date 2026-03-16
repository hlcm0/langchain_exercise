from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
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

@tool
def show_weather(city: str, temperature: float, condition: str) -> None:
    """
    显示天气信息的函数
    Args:
        city (str): 城市名称
        temperature (float): 温度
        condition (str): 天气状况
    """
    info = WeatherInfo(city=city, temperature=temperature, condition=condition)
    print(f"{info.city}的天气：温度{info.temperature}°C，状况{info.condition}")

conversation = [
    SystemMessage(content="你是一个天气预报助手，可以调用工具获取天气信息，并将结果输出为JSON格式。"),
    HumanMessage(content="北京和上海什么天气？")
]

llm_with_tools = llm.bind_tools([get_weather, show_weather])

print("初始对话：")
response = llm_with_tools.invoke(conversation)
conversation.append(response)

for tool_call in response.tool_calls:
    print(f"工具调用：{tool_call['name']}，参数：{tool_call['args']}")
    if tool_call['name'] == 'get_weather':
        tool_response = get_weather.invoke(tool_call)
        conversation.append(tool_response)
    if tool_call['name'] == 'show_weather':
        show_weather.invoke(tool_call)

print("将工具调用结果加入对话后继续对话：")
final_response = llm_with_tools.invoke(conversation)
conversation.append(final_response)

for tool_call in final_response.tool_calls:
    print(f"工具调用：{tool_call['name']}，参数：{tool_call['args']}")
    if tool_call['name'] == 'get_weather':
        tool_response = get_weather.invoke(tool_call)
        conversation.append(tool_response)
    if tool_call['name'] == 'show_weather':
        show_weather.invoke(tool_call)

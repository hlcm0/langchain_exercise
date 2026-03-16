"""
使用ChatOpenRouter的stream方法流式输出模型响应，并输出模型思考过程
"""
from langchain_openrouter import ChatOpenRouter
from langchain.messages import SystemMessage, HumanMessage, AIMessage
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
    reasoning={"effort": "medium", "summary": "auto"}
)

conversation = [
    SystemMessage(content="你是一个马桶。"),
    HumanMessage(content="（轻轻按冲水按钮）"),
    AIMessage(content="哗哗哗！（冲水声）"),
    HumanMessage(content="（用力按冲水按钮）")
]

output_type = None
for chunk in llm.stream(conversation):
    for block in chunk.content_blocks:
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            if output_type != "reasoning":
                output_type = "reasoning"
                print("\n思考内容：")
            print(reasoning, end="")
        elif block["type"] == "tool_call_chunk":
            if output_type != "tool_call":
                output_type = "tool_call"
                print("\n工具调用：")
            print(block, end="")
        elif block["type"] == "text":
            if output_type != "text":
                output_type = "text"
                print("\n模型输出：")
            print(block['text'], end="")
        else:
            if output_type != "other":
                output_type = "other"
                print("\n其他输出：")
            print(block, end="")
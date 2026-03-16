"""
使用 OpenAI SDK 直接调用 OpenRouter 上的模型，进行单轮对话并输出思考过程
"""
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

openrouter_model_name = os.getenv("OPENROUTER_MODEL_NAME")
if not openrouter_model_name:
    raise ValueError("OPENROUTER_MODEL_NAME is not set in the environment variables.")

client = OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "请介绍一下你自己。"}]
completion = client.chat.completions.create(
    model=openrouter_model_name,
    messages=messages,
    extra_body={"reasoning": {"effort": "medium", "summary": "auto"}},
    stream=True,
)

is_answering = False
print("思考过程：")
for chunk in completion:
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning", None)
    if reasoning:
        print(reasoning, end="", flush=True)

    reasoning_details = getattr(delta, "reasoning_details", None)
    if reasoning_details:
        for item in reasoning_details:
            if item.get("type") == "reasoning.text" and item.get("text"):
                print(item["text"], end="", flush=True)
            elif item.get("type") == "reasoning.summary" and item.get("summary"):
                print(item["summary"], end="", flush=True)

    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)
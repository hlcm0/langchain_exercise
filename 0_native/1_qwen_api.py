"""
使用OpenAI SDK直接调用Qwen模型的兼容接口，进行单轮对话并输出思考过程
"""
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

messages: list[ChatCompletionMessageParam]= [{"role": "user", "content": "请介绍一下你自己。"}]
completion = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=messages,
    extra_body={"enable_thinking": False},
    stream=True
)

is_answering = False
print("思考过程：")
for chunk in completion:
    delta = chunk.choices[0].delta
    reasoning_content = getattr(delta, "reasoning_content", None)
    if reasoning_content is not None:
        print(reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)
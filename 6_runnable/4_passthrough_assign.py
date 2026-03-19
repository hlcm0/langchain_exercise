"""
RunnablePassthrough.assign 示例。
保留原始输入，同时新增 answer 和 answer_length 两个字段。
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

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
    reasoning={"effort": "none", "summary": "auto"},
)

answer_chain = (
    ChatPromptTemplate.from_template("请简洁回答这个问题：{question}")
    | llm
    | StrOutputParser()
)

chain = RunnablePassthrough.assign(answer=answer_chain).assign(
    answer_length=lambda payload: len(payload["answer"])
)

response = chain.invoke({"question": "Runnable 在 LangChain 里是干什么用的？"})

print("原始问题：")
print(response["question"])
print("\n模型回答：")
print(response["answer"])
print("\n回答长度：")
print(response["answer_length"])
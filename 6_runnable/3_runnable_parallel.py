"""
RunnableParallel 示例。
让同一个输入同时走三条分支，生成摘要、翻译和关键词。
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
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

summary_chain = (
    ChatPromptTemplate.from_template("请用一句中文概括下面内容：{input}")
    | llm
    | StrOutputParser()
)

translation_chain = (
    ChatPromptTemplate.from_template("请把下面内容翻译成英文：{input}")
    | llm
    | StrOutputParser()
)

keywords_chain = (
    ChatPromptTemplate.from_template("请提取下面内容的三个关键词，用中文顿号分隔：{input}")
    | llm
    | StrOutputParser()
)

parallel_chain = RunnableParallel(
    summary=summary_chain,
    translation=translation_chain,
    keywords=keywords_chain,
)

response = parallel_chain.invoke(
    {"input": "江南皮革厂倒闭了，老板黄鹤带着他的小姨子跑了，我们没有办法，拿着钱包抵工资。原价都是三百多、二百多、一百多的钱包，现在全部只卖二十块，二十块钱你说便宜不便宜。快来买，快来买，走过路过千万不要错过。"}
)

print("摘要：")
print(response["summary"])
print("\n英文翻译：")
print(response["translation"])
print("\n关键词：")
print(response["keywords"])

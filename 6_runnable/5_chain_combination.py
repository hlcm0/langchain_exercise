"""
展示如何将多个 Runnable 链组合在一起，形成更复杂的数据处理流程。
输入字段 question、context和target_length。
输出字段 question、rewritten_question、answer、keywords、answer_length、missing_info、final_report
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

openrouter_model_name = os.getenv("OPENROUTER_MODEL_NAME")
if not openrouter_model_name:
    raise ValueError("OPENROUTER_MODEL_NAME is not set in the environment variables.")


input = {"question": "   \n 给出一个适合小明的运动计划。    ", 
         "context": "小明：男性，40岁，超重，从未锻炼过 小王：男性，30岁，体重正常，经常跑步", 
         "target_length": "简洁"}

def normalize_input(payload: dict) -> dict:
    cleaned_question = payload["question"].strip().replace("\n", " ")
    return {"question": payload["question"], "cleaned_question": cleaned_question, "context": payload["context"], "target_length": payload["target_length"]}


llm = ChatOpenRouter(
    model=openrouter_model_name,
    api_key=SecretStr(openrouter_api_key),
    reasoning={"effort": "none", "summary": "auto"},
)

rewrite_chain = (
    RunnableLambda(normalize_input)
    | ChatPromptTemplate.from_template(
        "请根据以下背景信息重写这个问题，使其更清晰、更具体，仅给出重写后的问题的一个版本：\n\n背景信息：{context}\n\n问题：{cleaned_question}"
    )
    | llm
    | StrOutputParser()
)

answer_chain = (
    ChatPromptTemplate.from_template(
        "你是一个聊天助手，你需要回答用户的以下问题，回答要{target_length}： {rewritten_question}"
    )
    | llm
    | StrOutputParser()
)

keywords_chain = (
    ChatPromptTemplate.from_template(
        "请提取回答中的三个关键词，用中文顿号分隔：{answer}"
    )
    | llm
    | StrOutputParser()
)

missing_info_chain = (
    ChatPromptTemplate.from_template(
        "请分析这个问题是否缺乏关键信息，如果缺乏，请列出缺乏的信息点，长度{target_length}：{rewritten_question}"
    )
    | llm
    | StrOutputParser()
)

final_report_chain = (
    ChatPromptTemplate.from_template(
        "请根据以下内容生成一个最终报告，长度{target_length}：问题：{rewritten_question}\n答案：{answer}\n关键词：{keywords}\n缺失信息：{missing_info}"
    )
    | llm
    | StrOutputParser()
)

chain = RunnablePassthrough \
    .assign(rewritten_question=rewrite_chain) \
    .assign(answer=answer_chain, missing_info=missing_info_chain) \
    .assign(keywords=keywords_chain) \
    .assign(final_report=final_report_chain) \
    .assign(answer_length=lambda payload: len(payload["answer"]))

response = chain.invoke(input)

print(response)
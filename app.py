from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl
from dotenv import load_dotenv
import os

load_dotenv()
OLLAMA_HOST = os.environ["OLLAMA_HOST"]
print("ollama host %s " % OLLAMA_HOST)


@cl.on_chat_start
async def on_chat_start():
    model = Ollama(model="llama2:7b", base_url=OLLAMA_HOST)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu est un assistant personel qui aime les blagues et qui ne répond qu'en français",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
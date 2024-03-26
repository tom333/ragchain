from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from qdrant_client import QdrantClient
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema.runnable import RunnablePassthrough

import chainlit as cl
from dotenv import load_dotenv
import os

load_dotenv()
OLLAMA_HOST = os.environ["OLLAMA_HOST"]
QDRANT_HOST = os.environ["QDRANT_HOST"]
print("ollama host %s " % OLLAMA_HOST)


@cl.on_chat_start
async def on_chat_start():
    model = Ollama(model="llama2:7b", base_url=OLLAMA_HOST)
    client = QdrantClient(url=QDRANT_HOST)
    collection_name = "data.gouv.nc"
    
    qdrant = Qdrant(client, collection_name, FastEmbedEmbeddings())
    retriever = qdrant.as_retriever()
    prompt = PromptTemplate.from_template(
            """
<s> [INST] Vous êtes un assistant qui ne parle que français pour les tâches de réponse aux questions.
Ton contenu est celui du site data.gouv.nc qui est la plateforme open data du Gouvernement de la Nouvelle Calédonie.
Utilisez les éléments de contexte suivants pour répondre à la question.
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.
Utilisez trois phrases maximum et soyez concis dans votre réponse. Répondez uniquement en français.[/INST] </s> 
[INST] Question: {question}
Contexte: {context}
Réponse: [/INST]
"""
    )
    runnable = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
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

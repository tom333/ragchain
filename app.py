from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.chains.llm import LLMChain
from langchain.schema.runnable.config import RunnableConfig
from qdrant_client import QdrantClient
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import chainlit as cl
from dotenv import load_dotenv
import os
from pprint import pprint

load_dotenv()
OLLAMA_HOST = os.environ["OLLAMA_HOST"]
QDRANT_HOST = os.environ["QDRANT_HOST"]
print("ollama host %s " % OLLAMA_HOST)

# model_name = "llama2:7b"
model_name = "mistral"
model = Ollama(model=model_name, base_url=OLLAMA_HOST)
client = QdrantClient(url=QDRANT_HOST)
collection_name = "data.gouv.nc"

qdrant = Qdrant(client, collection_name, FastEmbedEmbeddings())
retriever = qdrant.as_retriever()


async def summurize_pdf():
    files = None
    while not files:
        files = await cl.AskFileMessage(
            content="Upload du PDF",
            accept=["application/pdf"],
            max_size_mb=10,
            timeout=180,
        ).send()

    file = files[0]


    processing_msg = cl.Message(
        content=f"Génération du résumé du document `{file.name}`...",
        disable_feedback=True
    )
    await processing_msg.send()

    pdf_loader = PyPDFLoader(file_path=file.path)
    docs = pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
    print("pdf chargé")
    prompt_template = """Rédige en français un court résumé du document suivant:
                    {text}
                    Résumé en français :"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=model, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    
    print("chain générée")
    msg = cl.Message(content="")

    async for chunk in stuff_chain.astream(
        {"text": docs, "input_documents": docs},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):  
        pprint(chunk)
        await msg.stream_token(chunk["output_text"])

    await msg.send()
    
    
    

@cl.on_chat_start
async def on_chat_start():
    
    prompt = ChatPromptTemplate.from_template(
            """
<s> [INST] Vous êtes un assistant qui ne parle que français pour les tâches de réponse aux questions.
Ton contenu est celui du site data.gouv.nc qui est la plateforme open data du Gouvernement de la Nouvelle Calédonie.
Chaque document du contexte est un jeux de données.
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
    
    res = await cl.AskActionMessage(
        content="Que voulez vous faire ?",
        actions=[
            cl.Action(name="Chercher un jeu de données", value="JDD", label="Chercher un jeu de données"),
            cl.Action(name="Résumer un document", value="RAG", label="Résumer un document"),
        ],
    ).send()

    if res and res.get("value") == "JDD":
        await cl.Message(
            content="Quel jeu de données cherchez vous ?",
        ).send()
        
    elif res and res.get("value") == "RAG":        
        await summurize_pdf()


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

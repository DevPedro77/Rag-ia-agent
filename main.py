import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PASTA_PDFS = "dados/pdfs"
PASTA_CHROMA = "chroma_db"
MODELO_LLM = "gpt-3.5-turbo"

# Carrega os PDFs
def carregar_pdfs():
    documentos = []
    for nome in os.listdir(PASTA_PDFS):
        if nome.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PASTA_PDFS, nome), extract_images=False)
            documentos.extend(loader.load())
    return documentos

#Separa em chunks (Pedacoes de Documentos)
def dividir_documentos(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=20,
        length_function = len,
        add_start_index = True,
    )
    return splitter.split_documents(documentos)

# Criar o embbeding
def criar_embeddings():
    return OpenAIEmbeddings()

#Salvar no vector DB
def criar_vectorstore(chunks, embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PASTA_CHROMA,
    )

#
def carregar_vectorstore(embeddings):
    return Chroma(
        embedding_function=embeddings,
        persist_directory=PASTA_CHROMA,
    )


def criar_llm():
    return ChatOpenAI(model=MODELO_LLM, max_tokens = 300)


def criar_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Responda somente com base no contexto abaixo.\n\n{contexto}"),
        ("human", "{pergunta}"),
    ])


def formatar_contexto(documentos):
    return "\n\n".join(doc.page_content for doc in documentos)


def responder(pergunta, vectorstore, llm, prompt, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    trechos = retriever.invoke(pergunta)

    cadeia = prompt | llm | StrOutputParser()
    return cadeia.invoke({
        "contexto": formatar_contexto(trechos),
        "pergunta": pergunta,
    })


def indexar():
    documentos = carregar_pdfs()
    chunks = dividir_documentos(documentos)
    embeddings = criar_embeddings()
    return criar_vectorstore(chunks, embeddings)


if __name__ == "__main__":
    if os.path.exists(PASTA_CHROMA):
        db = carregar_vectorstore(criar_embeddings())
    else:
        db = indexar()

    modelo = criar_llm()
    template = criar_prompt()

    entrada = input("Pergunta: ")
    saida = responder(entrada, db, modelo, template)
    print(saida)

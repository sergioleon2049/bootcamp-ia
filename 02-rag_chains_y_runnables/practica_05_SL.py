from dotenv import load_dotenv
import os

#Cargar variables de entorno
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

#Ejercicio 1

#Instanciar PDF Loader
loader = PyPDFLoader("data/BOE-A-2023-17238.pdf")

# Instanciar Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

#Lo divide en fragmentos
docs = loader.load_and_split(text_splitter=splitter)

#Mostrar número total de documentos
print(f"Número total de documentos: {len(docs)}")

# Mostrar primer documento
print(docs[0])

# Mostrar página y contenido de los 3 primeros documentos
for doc in docs[:3]:
    page = doc.metadata.get("page", "N/A")
    contenido = doc.page_content[:100]
    print(f"Página: {page}")
    print(f"Contenido: {contenido}")
    print("-" * 50)

# Ejercicio 2

# Instanciar Embeddings model
#Convierte texto en números
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

#Instanciar VectorStore
vectorstore = PineconeVectorStore(
    index_name="boe-index",
    embedding=embeddings
)

# Añadir documentos a la VectorStore
vectorstore.add_documents(docs)
print(f"Documentos añadidos a Pinecone: {len(docs)}")

#Record count de 182 en pinecone, comprobado

#Ejercicio 3
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

#Instanciar Retriever con top-k=2
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#Realizar una consulta sobre vacaciones
resultados = retriever.invoke("¿Cuántos días de vacaciones tienen los trabajadores?")

#Mostrar números de página de los documentos recuperados
for doc in resultados:
    print(f"Página: {doc.metadata.get('page', 'N/A')}")

#Ejercicio 4
#Instanciar LLM
llm = ChatOpenAI(model="gpt-5-mini")

#Definir prompt base
prompt = ChatPromptTemplate.from_template("""
Responde a la pregunta utilizando únicamente el siguiente contexto.
Si no encuentras la respuesta en el contexto, di que no lo sabes.

Contexto: {context}

Pregunta: {question}
""")

#Definir chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

#Query y respuesta
response = chain.invoke("Cuántos días de vacaciones tienen los trabajadores?")
print(response.content)

#Modificar prompt para incluir número de página
prompt_con_pagina = ChatPromptTemplate.from_template("""
Responde a la pregunta utilizando únicamente el siguiente contexto.
Menciona el número de página donde encontraste la información relevante.
Si no encuentras la respuesta en el contexto, di que no lo sabes.

Contexto: {context}

Pregunta: {question}
""")

#Definir chain con nuevo prompt
chain_con_pagina = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_con_pagina
    | llm
)

#Query y respuesta con página
response2 = chain_con_pagina.invoke("Cuántos días de vacaciones tienen los trabajadores?")
print(response2.content)
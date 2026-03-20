import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

#Cargamos los documentos

carpeta= "documentos_soporte_tecnico"
documentos= []

for archivo in os.listdir(carpeta):
    if archivo.endswith(".txt"):
        ruta= os.path.join(carpeta, archivo)
        loader = TextLoader(ruta, encoding= 'utf-8')
        documentos.extend(loader.load())

print(f"Documentos cargados: {len(documentos)}")

#Split

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fragmentos = splitter.split_documents(documentos)
print(f"Fragmentos generados: {len(fragmentos)}")

#Embeddings

embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
vectorstore= FAISS.from_documents(fragmentos, embeddings)

#Retriever

retriever= vectorstore.as_retriever(search_kwargs={"k": 3})

#Instanciamos el modelo LLM

llm= ChatOpenAI(
    model='gpt-5-mini',
    temperature=1
)

#Prompt template

prompt= ChatPromptTemplate.from_template("""
    Eres un asistente técnico especializado en maquinaria industrial.
    Responde la pregunta usando únicamente el siguiente contexto.
    Si no encuentras la respuesta, di que no tienes esa información.

    Contexto: {context}
                                         
    Pregunta: {question}
""")

#RAG Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Bucle de preguntas
print("\nSistema RAG iniciado. Escribe 'salir' para terminar.\n")

while True:
    pregunta = input("Técnico: ")
    
    if pregunta.lower() == "salir":
        print("Sistema cerrado.")
        break
    
    response = chain.invoke(pregunta)
    print(f"\nAsistente: {response.content}\n")
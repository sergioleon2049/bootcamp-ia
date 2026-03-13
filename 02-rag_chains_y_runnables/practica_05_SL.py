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
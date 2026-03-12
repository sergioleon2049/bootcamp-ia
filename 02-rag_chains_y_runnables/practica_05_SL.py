from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Instanciar PDF Loader
loader = PyPDFLoader("data/BOE-A-2023-17238.pdf")

#Instanciar Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

#Load and split PDF
docs = loader.load_and_split(text_splitter=splitter)

#Mostrar número total de documentos
print(f"Número total de documentos: {len(docs)}")

#Mostrar primer documento
print(docs[0])

#Mostrar pagina y contenido de los 3 primeros documentos
for doc in docs[:3]:
    page = doc.metadata.get("page", "N/A")
    contenido = doc.page_content[:100]
    print(f"Página: {page}")
    print(f"Contenido: {contenido}")
    print("-" * 50)
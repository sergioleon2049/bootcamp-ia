import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

#el modulo de pymupdf se llama fitz
import fitz
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Instanciar LLM
llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

# Definir parser JSON
parser = JsonOutputParser()

# Definir prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Eres un experto en análisis de contratos de seguros.
    Extrae la siguiente información del contrato y devuelve ÚNICAMENTE un JSON válido con estos campos:
    - nombre_asegurado
    - tipo_cobertura
    - fecha_inicio
    - fecha_fin
    - prima_anual
    Si algún campo no aparece en el contrato, pon null.
    No añadas explicaciones, solo el JSON.
    """),
    ("human", "{texto}")
])

#Definir chain
chain = prompt | llm | parser

#Función para extraer texto de un PDF
def extraer_texto_pdf(ruta):
    doc = fitz.open(ruta)
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto

# Procesar todos los PDFs de la carpeta
carpeta = "pdf_contratos_seguro"
resultados = []

for archivo in os.listdir(carpeta):
    if archivo.endswith(".pdf"):
        ruta = os.path.join(carpeta, archivo)
        print(f"Procesando: {archivo}")
        
        try:
            # Extraer texto del PDF
            texto = extraer_texto_pdf(ruta)
            
            # Invocar la chain
            datos = chain.invoke({"texto": texto})
            datos["archivo"] = archivo
            resultados.append(datos)
            
        except Exception as e:
            print(f"Error en {archivo}: {e}")
            resultados.append({
                "nombre_asegurado": None,
                "tipo_cobertura": None,
                "fecha_inicio": None,
                "fecha_fin": None,
                "prima_anual": None,
                "archivo": archivo
            })

# Crear DataFrame
df = pd.DataFrame(resultados)
print(df)

#Opcional: crear un gráfico

import matplotlib.pyplot as plt

#Limpiar columna prima_anual para que sea numérica
df["prima_anual_num"] = df["prima_anual"].astype(str).str.extract(r'(\d+)').astype(float)

#Gráfico 1: Distribución de tipos de cobertura
plt.figure(figsize=(8, 5))
df["tipo_cobertura"].value_counts().plot(kind="bar", color="steelblue")
plt.title("Distribución de tipos de cobertura")
plt.xlabel("Tipo de cobertura")
plt.ylabel("Número de contratos")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Gráfico 2: Prima anual promedio por tipo de cobertura
plt.figure(figsize=(8, 5))
df.groupby("tipo_cobertura")["prima_anual_num"].mean().plot(kind="bar", color="darkorange")
plt.title("Prima anual promedio por tipo de cobertura")
plt.xlabel("Tipo de cobertura")
plt.ylabel("Prima anual media (€)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
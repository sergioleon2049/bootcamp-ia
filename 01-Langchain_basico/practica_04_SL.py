#Importamos parser

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser

#Cargando el env
load_dotenv()

#Instanciamos el parser
parser= SimpleJsonOutputParser()

#Instanciamos el modelo
llm=ChatOpenAI(
    model="gpt-5-mini",
    temperature=1
)

#Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Genera un reporte financiero ficticio en formato JSON.

    El JSON debe tener exactamente estas claves:
    - ingresos
    - gastos
    - beneficio_neto

    Los valores deben ser números.

    Empresa: {empresa}
    """)
])

#Creamos cadena

chain= prompt | llm | parser

response= chain.invoke({
    "empresa": "Panadería Luis"
})

print(response)
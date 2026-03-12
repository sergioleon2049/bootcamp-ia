from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# Cargar variables de entorno
load_dotenv()

# Instanciar modelo, no hay soporte de temp en gpt5mini
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1
)


# Prompt
messages = [
    SystemMessage(content="Eres un profesor de biología"),
    HumanMessage(content="¿Cómo funciona la fotosíntesis?")
]


# Respuesta del modelo
response = llm.invoke(messages)


# Mostrar respuesta
print(response.content)
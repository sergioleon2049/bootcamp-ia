#Probando: usar streams en la respuesta(PENDIENTE)

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
#from langchain.callbacks.base import CallbackManager, BaseCallbackHandler

# cargar variables de entorno
load_dotenv()

# modelo
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1
)

#Prompt template
template = """
Escribe una review {sentimiento} sobre el siguiente producto:

Producto: {producto}

La review debe tener 2 o 3 frases.
"""

prompt = PromptTemplate(
    input_variables=["producto", "sentimiento"],
    template=template
)

#Producto 1
prompt_formatted = prompt.format(
    producto="iPhone 15",
    sentimiento="positiva"
)

response = llm.invoke(prompt_formatted)

print("Review producto 1:")
print(response.content)
print()

#Producto 2
prompt_formatted = prompt.format(
    producto="teclado mecánico",
    sentimiento="negativa"
)

response = llm.invoke(prompt_formatted)

print("Review producto 2:")
print(response.content)
print()

#Producto que no existe
prompt_formatted = prompt.format(
    producto="MegaPhone 9",
    sentimiento="positiva"
)

response = llm.invoke(prompt_formatted)

print("Review producto inexistente:")
print(response.content)

#Cuando le otorgamos un producto inexistente, nos encontramos con alucinaciones
#Funciona por probabilidad de palabras
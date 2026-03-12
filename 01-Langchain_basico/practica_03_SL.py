from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

#Cargando el env
load_dotenv()

#Instanciamos el modelo
llm=ChatOpenAI(
    model="gpt-5-mini",
    temperature=1
)

#Prompt Template

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Eres un asistente de recursos humanos en una empresa.
Un trabajador tiene dudas sobre la política de trabajo en remoto.
Debes responder de forma clara, concisa y profesional.
"""),
    ("human", "{pregunta}")
])

#Usamos LCEL para crear la cadena
chain = prompt | llm

response= chain.invoke({"pregunta":"Cuántos días a la semana puedo trabajar de forma remota?"})
response2= chain.invoke({"pregunta":"me puedo ir a otro país los días en remoto?"})

print(response.content)
print(response2.content)
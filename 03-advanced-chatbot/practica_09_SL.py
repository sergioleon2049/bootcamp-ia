import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage

#instanciamos el chatbot

chatbot = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1
    )

#Memoria en RAM, no persistente
memory = ConversationBufferMemory(
    memory_key="messages",
    return_messages=True
)

#Prompt con system message y historial
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        SystemMessage(content="""
        Eres un chatbot de atención al cliente en un banco.
        Recibirás preguntas sobre productos financieros. Sé conciso.
        El banco ofrece cuentas corrientes, tarjetas de crédito, préstamos personales e hipotecas.
        Para abrir una cuenta se necesita: DNI, comprobante de domicilio e ingreso mínimo de 500€.
        Las tarjetas de crédito tienen una cuota anual de 30€, y un límite de crédito estándar de 2000€.
        Los préstamos personales tienen un interés del 6% TIN y se pueden solicitar hasta 20.000€.
        Las hipotecas tienen un interés del 3,5% TIN a 30 años y requieren entrada del 20%.
        Si te preguntan algo fuera de tu conocimiento, responde que no tienes esa información.
        """),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

#Chain con memoria
chain = LLMChain(
    llm=chatbot,
    prompt=prompt,
    memory=memory
)

print("Chatbot iniciado. Escribe 'salir' para terminar.\n")

#Bucle conversacional
while True:
    user_input = input("Usuario: ")
    
    if user_input.lower() == "salir":
        print("Bot: ¡Hasta luego!")
        break
    
    response = chain.invoke(user_input)
    print(f"Bot: {response['text']}\n")
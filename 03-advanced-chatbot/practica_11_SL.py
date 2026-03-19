import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Instanciar llm
llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

#prompt

prompt = ChatPromptTemplate([
    ("system", """
    Eres un analista de sentimientos especializado en e-commerce.
    Clasifica el sentimiento del comentario como exactamente una de estas palabras: positivo, negativo o neutro.
    No añadas explicaciones, solo devuelve una de las tres palabras.
     """),
     ("human", "{comentario}")
])

#definir cadena
chain = prompt | llm | StrOutputParser()

#Cargar CSV
df = pd.read_csv("data_reviews/data_reviews.csv")

#clasificar cada sentimiento
print("Analizando sentimientos... \n")
df["sentimiento"] = df["Comentario"].apply(lambda x: chain.invoke({"comentario": x}))

print(df)

#gráfico opcional
import matplotlib.pyplot as plt

# Gráfica de barras con total de reseñas por sentimiento
df["sentimiento"].value_counts().plot(kind="bar", color=["green", "red", "gray"])
plt.title("Total de reseñas por sentimiento")
plt.xlabel("Sentimiento")
plt.ylabel("Número de reseñas")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
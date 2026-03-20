import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
_= load_dotenv(find_dotenv())
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Instanciar LLM
llm = ChatOpenAI(model='gpt-5-mini', temperature=1)
sql_parser = StrOutputParser()

#Conectar a la BBDD
db = SQLDatabase.from_uri("sqlite:///football_db.sqlite")

#Metadata info
metadata_text = """
La base de datos contiene las siguientes tablas relevantes:
Match:
- Datos de cada partido: fecha (`date`), temporada (`season`), goles (`home_team_goal`, `away_team_goal`), equipos (`home_team_api_id`, `away_team_api_id`) y alineaciones (`home_player_1` a `home_player_11`, etc.).
Team:
- Equipos con `team_api_id` y nombre largo `team_long_name`.
Player:
- Jugadores con `player_api_id`, `player_name`, `birthday`, `height`, `weight`.
League:
- Ligas: `league_id`, `country_id`, `name`.
Country:
- Países, con campo `name`.
Relaciones clave:
- Match.home_team_api_id ↔ Team.team_api_id
- Match.away_team_api_id ↔ Team.team_api_id
- Match.home_player_N ↔ Player.player_api_id
- League.country_id ↔ Country.id
Notas:
- Los goles se registran en `home_team_goal` y `away_team_goal`.
- Las temporadas están formateadas como `'2008/2009'`, `'2011/2012'`, etc.
"""

# Chain 1: pregunta a SQL
sql_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
    Eres un experto en bases de datos de fútbol.
    Dado una pregunta del usuario, responde ÚNICAMENTE con una consulta SQL válida y ejecutable sobre una base SQLite.
    No incluyas explicaciones, comentarios ni bloques markdown.
    NO uses instrucciones que modifiquen los datos (como INSERT, UPDATE, DELETE).
    Cruza siempre contra las tablas de dimensiones si es necesario para que la respuesta sea informativa.
    Usa solo las siguientes tablas (ya existentes en la base de datos):
    {db.get_table_info()}
    Aquí tienes información adicional para entender la estructura de las tablas y sus relaciones:
    {metadata_text}
    Devuelve únicamente la consulta SQL, sin texto adicional.
    """),
    ("user", "{pregunta}")
])
sql_chain = sql_prompt | llm | sql_parser

#Chain 2: resultado SQL. Respuesta en lenguaje natural
respuesta_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Eres un asistente experto en fútbol.
    Dado el resultado de una consulta SQL, responde a la pregunta del usuario en lenguaje natural claro y conciso.
    """),
    ("user", "Pregunta: {pregunta}\nResultado SQL: {resultado}")
])
respuesta_chain = respuesta_prompt | llm | sql_parser

#Bucle de preguntas
print("Sistema de consultas de fútbol habilitado. Escribe salir para terminar.\n")

while True:
    pregunta = input("Usuario: ")
    if pregunta.lower() == "salir":
        print("Apagando...")
        break
    try:
        #Generar SQL
        query = sql_chain.invoke({"pregunta": pregunta})
        print(f"\nSQL generado: \n{query}\n")

        #Ejecutar query
        resultado = db.run(query)
        print(f"Resultado SQL: \n{resultado}\n")

        #Mostrar resultado como DataFrame si es posible
        df_resultado = None
        try:
            import ast
            datos = ast.literal_eval(resultado)
            if isinstance(datos, list) and len(datos) > 0:
                df_resultado = pd.DataFrame(datos)
                print("Resultado en tabla:")
                print(df_resultado.to_string(index=False))
                print()
        except:
            pass

        #Respuesta en lenguaje natural
        respuesta = respuesta_chain.invoke({"pregunta": pregunta, "resultado": resultado})
        print(f"Respuesta: {respuesta}\n")

        #Ofrecer gráfico si hay datos numéricos
        if df_resultado is not None and not df_resultado.empty:
            cols_numericas = df_resultado.select_dtypes(include='number').columns.tolist()
            if cols_numericas:
                ver_grafico = input("¿Quieres ver un gráfico de estos resultados? (s/n): ")
                if ver_grafico.lower() == "s":
                    col_numerica = cols_numericas[-1]
                    col_etiqueta = df_resultado.columns[0]
                    plt.figure(figsize=(10, 5))
                    plt.bar(df_resultado[col_etiqueta].astype(str), df_resultado[col_numerica], color="steelblue")
                    plt.title(pregunta)
                    plt.xlabel(col_etiqueta)
                    plt.ylabel(col_numerica)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.show()

    except Exception as e:
        print(f"Error al procesar la consulta: {e}\n")
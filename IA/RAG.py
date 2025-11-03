import csv
import os
from pathlib import Path

import openai
# from dotenv import load_dotenv
from lunr import lunr

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
token = os.environ["GITHUB_TOKEN"]
MODEL_NAME = os.getenv("GITHUB_MODEL", model)

client = openai.OpenAI(
    base_url= endpoint,
    api_key= token
)



# Indexamos los datos del CSV
CSV_PATH = Path(__file__).resolve().parent.parent / "CSV" / "hybridos.csv"
with CSV_PATH.open(newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    rows = list(reader)

documents = [{"id": (i + 1), "body": " ".join(row)} for i, row in enumerate(rows[1:])]
index = lunr(ref="id", fields=["body"], documents=documents)

# Obteneemos la pregunta del usuario
user_question = "¿Cual el mejor Prius considerando sus caracteristica y precio?"

# Buscamos en el índice la pregunta del usuario
results = index.search(user_question)
matching_rows = [rows[int(result["ref"])] for result in results]


# Formateamos como tabla markdown, ya que los llms entienden markdown
matches_table = " | ".join(rows[0]) + "\n" + " | ".join(" --- " for _ in range(len(rows[0]))) + "\n"
matches_table += "\n".join(" | ".join(row) for row in matching_rows)



# Ahora podemos usar los resultados para generar una respuesta
SYSTEM_MESSAGE = """
Eres un asistente útil que responde preguntas sobre automóviles basándote en un conjunto de datos de autos híbridos.
Debes utilizar el conjunto de datos para responder las preguntas, no debes
proporcionar ninguna información que no esté en las fuentes proporcionadas.
"""

print(f"{user_question}\nSources: {matches_table}")

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.3,
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{user_question}\nSources: {matches_table}"},
    ],
)


print(f"\nRespuest de github: \n")
print(response.choices[0].message.content)


import json
import os
import openai
from lunr import lunr

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
model_embedding = "text-embedding-3-small"
token = os.environ["GITHUB_TOKEN"]
MODEL_NAME = os.getenv("GITHUB_MODEL", model)

client = openai.OpenAI(
    base_url= endpoint,
    api_key= token
)

# Indexar los datos del JSON - cada objeto tiene id, texto y embedding
with open("Vectores/rag_ingested_chunks.json") as file:
    documents = json.load(file)
    documents_by_id = {doc["id"]: doc for doc in documents}
index = lunr(ref="id", fields=["text"], documents=documents)

# Obtener la pregunta del usuario
user_question = "¿como se llama la abeja doméstica?"

# Buscar la pregunta del usuario en el índice
results = index.search(user_question)
retrieved_documents = [
    documents_by_id[result["ref"]]
    for result in results
    if result["score"] >= 2 and result["ref"] in documents_by_id
    ]


print(f"{len(retrieved_documents)} documentos recuperados coincidentes mayor a 2.")
context = "\n".join([f"{doc['id']}: {doc['text']}" for doc in retrieved_documents])

print("--------------------------------------------------------------------------------------------------------------------------------")
print(context)
print("--------------------------------------------------------------------------------------------------------------------------------")

# Ahora podemos usar las coincidencias para generar una respuesta
SYSTEM_MESSAGE = """
Eres un asistente útil que responde preguntas sobre insectos regionales.
Debes utilizar el conjunto de datos para responder las preguntas,
no debes proporcionar ninguna información que no esté en las fuentes proporcionadas.
Cita las fuentes que utilizaste para responder la pregunta entre corchetes.
Las fuentes están en el formato: <id>: <texto>.
"""

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.3,
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{user_question}\nFuentes: {context}"},
    ],
)

print(f"\nRespuesta: \n")
print(response.choices[0].message.content)
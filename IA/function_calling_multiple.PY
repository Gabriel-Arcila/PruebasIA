import os
import openai
import base64
import json

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
model_embedding = "text-embedding-3-small"
token = os.environ["GITHUB_TOKEN"]
MODEL_NAME = os.getenv("GITHUB_MODEL", model)

client = openai.OpenAI(
    base_url= endpoint,
    api_key= token
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Busca el clima según nombre de ciudad o código postal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "El nombre de la ciudad",
                    },
                    "zip_code": {
                        "type": "string",
                        "description": "El código postal",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_movies",
            "description": "Buscar películas en cines según nombre de ciudad o código postal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "El nombre de la ciudad",
                    },
                    "zip_code": {
                        "type": "string",
                        "description": "El código postal",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Eres un chatbot de turismo."},
        {
            "role": "user",
            "content": "¿Está lloviendo lo suficiente en Sídney como para ver películas y cuáles estan en los cines?",
        },
    ],
    tools=tools,
    tool_choice="auto",
)

print(f"Respuesta: \n")
for message in response.choices[0].message.tool_calls:
    print(message.function.name)
    print(message.function.arguments)
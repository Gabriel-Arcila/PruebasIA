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

def lookup_weather(city_name=None, zip_code=None):
    """Buscar el clima para un nombre de ciudad o código postal dado."""
    print(f"Buscando el clima para {city_name or zip_code}...")
    return "¡Está soleado!"

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Buscar el clima para un nombre de ciudad o código postal dado.",
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
                "strict": True,
                "additionalProperties": False,
            },
        },
    }
]

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Eres un chatbot del clima."},
        {"role": "user", "content": "¿está soleado en esa pequeña ciudad cerca de Sydney donde vive Anthony?"},
    ],
    tools=tools,
    tool_choice="auto",
)


print(f"Respuesta de {MODEL_NAME}: \n")

print(response.choices[0].message.content)
print()
print(response.choices[0].message.tool_calls)
print()


if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    if function_name == "lookup_weather":
        print(lookup_weather(**arguments))


print("\n---------------------------\n")

messages = [
    {"role": "system", "content": "Eres un chatbot de clima."},
    {"role": "user", "content": "Está soleado en Berkeley CA?"},
]
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(f"Respuesta de {MODEL_NAME}: \n")


# Ahora llama a la función indicada

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "lookup_weather":
        messages.append(response.choices[0].message)
        result = lookup_weather(**arguments)
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
        print(response.choices[0].message.content)
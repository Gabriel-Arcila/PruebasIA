import os
import openai
import base64


endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
model_embedding = "text-embedding-3-small"
token = os.environ["GITHUB_TOKEN"]
MODEL_NAME = os.getenv("GITHUB_MODEL", model)

client = openai.OpenAI(
    base_url= endpoint,
    api_key= token
)


messages = [
    {
        "role": "user",
        "content": [
            {"text": "Esto es un unicornio?", "type": "text"},
            {
                "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Ur-painting.jpg"},
                "type": "image_url",
            },
        ],
    }
]
response = client.chat.completions.create(model=model, messages=messages, temperature=0.5)

print(response.choices[0].message.content)
print()

def open_image_as_base64(filename):
    with open(filename, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "esos son cocodrilos o caimanes?", "type": "text"},
                {"image_url": {"url": open_image_as_base64("img/mystery_reptile.png")}, "type": "image_url"},
            ],
        }
    ],
)

print(response.choices[0].message.content)
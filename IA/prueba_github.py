import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(""),
        UserMessage("Explicame que es la IA, de una manera corta y sin Markdown"),
    ],
    model=model,
    temperature=1,
    #n =1
    
)

print(response.choices[0].message.content)


completion = client.complete(
    messages=[
        SystemMessage("""
                    Eres un ingeniero profecional con altas experiencias en tecnologia, 
                    explicame de forma tecnica y detallada,
                    no me coloques caracteres especiales de Markdown
                    """),
        UserMessage("Explicame que es la IA"),
    ],
    model=model,
    temperature=1,
    #n =1
    max_tokens=1024,
    stream=True
)

print("\n \n")

for event in completion:
    if event.choices:
        content = event.choices[0].delta.content
        if content:
            print(content, end="", flush=True)


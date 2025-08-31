from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explicame que es la IA, de una manera corta y sin Markdown",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    )
)
print(response.text)


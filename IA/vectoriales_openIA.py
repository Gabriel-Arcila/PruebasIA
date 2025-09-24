import os
import openai


endpoint = "https://models.github.ai/inference"
model = "openai/text-embedding-3-small"
token = os.environ["GITHUB_TOKEN"]
dimensions = 1536

openai_client = openai.OpenAI(
    base_url= endpoint,
    api_key= token
)

embeddings_response = openai_client.embeddings.create(
    model=model,
    dimensions= dimensions,
    input="Hola Mundo",
)
embedding = embeddings_response.data[0].embedding

print(len(embedding))
#print(embedding)

import json
import os
import pathlib
import openai

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
model_embedding = "text-embedding-3-small"
token = os.environ["GITHUB_TOKEN"]
MODEL_NAME = os.getenv("GITHUB_MODEL", model)

client = openai.OpenAI(
    base_url= endpoint,
    api_key= token
)


#data_dir = pathlib.Path(os.path.dirname(__file__)) / "Pdf"
data_dir = pathlib.Path(__file__).resolve().parent.parent / "Pdf"
filenames = ["Xylocopa_californica.pdf", "Centris_pallida.pdf", "Apis_mellifera.pdf", "Syrphidae.pdf"]
all_chunks = []

for filename in filenames:
    # Extraemos texto del archivo PDF
    md_text = pymupdf4llm.to_markdown(data_dir / filename)

    # Dividimos el texto en fragmentos más pequeños
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4.1", chunk_size=500, chunk_overlap=125
    )
    texts = text_splitter.create_documents([md_text])
    file_chunks = [{"id": f"{filename}-{(i + 1)}", "text": text.page_content} for i, text in enumerate(texts)]

    # Generamos embeddings utilizando el SDK de openAI para cada texto
    for file_chunk in file_chunks:
        file_chunk["embedding"] = (
            client.embeddings.create(model=model_embedding, input=file_chunk["text"]).data[0].embedding
        )
    all_chunks.extend(file_chunks)

# Guardamos los documentos con embeddings en un archivo JSON
with open("rag_ingested_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=4)
import json
import os
import openai
import pandas as pd
import hnswlib

with open('Vectores/peliculas_text-embedding-3-small-1536.json') as f:
    movies = json.load(f)

endpoint = "https://models.github.ai/inference"
model = "openai/text-embedding-3-small"
token = os.environ["GITHUB_TOKEN"]
dimensions = 1536

openai_client = openai.OpenAI(
    base_url= endpoint,
    api_key= token
)

def get_embedding(text):
    embeddings_response = openai_client.embeddings.create(
        model=model,
        dimensions=dimensions,
        input=text,
    )
    return embeddings_response.data[0].embedding

#*Busqueda de embeddings vectorial

def cosine_similarity(v1, v2):
    dot_product = sum([a * b for a, b in zip(v1, v2)])
    magnitude = (sum([a**2 for a in v1]) * sum([a**2 for a in v2])) ** 0.5
    return dot_product / magnitude

def exhaustive_search(query_vector, vectors):
    similarities = []
    for title, vector in vectors.items():
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((title, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

new_vector = get_embedding("una pelicula que trate sobre la busque de lugares escondidos debajo del agua")
similarities = exhaustive_search(new_vector, movies)
most_similar = similarities[0:10]
similar_movies = [(movie, round(similarity, 3)) for movie, similarity in most_similar]

print(pd.DataFrame(similar_movies, columns=['pelicula', 'similitud']))

#* Búsqueda ANN: HNSW

# Declarando índice
p = hnswlib.Index(space='cosine', dim=1536)

# Inicializando índice - el número máximo de elementos debe conocerse de antemano
p.init_index(max_elements=len(movies), ef_construction=200, M=16)

# Inserción de elementos (puede ser llamada varias veces):
vectors = list(movies.values())
ids = list([i for i in range(len(vectors))])
p.add_items(vectors, ids)

# Controlando la recuperación mediante el ajuste de ef:
p.set_ef(50) # ef siempre debe ser > k

print()
print()

### Los parámetros del índice se exponen como propiedades de clase:
print(f"Parámetros pasados al constructor:  space={p.space}, dim={p.dim}") 
print(f"Construcción del índice: M={p.M}, ef_construction={p.ef_construction}")
print(f"El tamaño del índice es {p.element_count} y la capacidad del índice es {p.max_elements}")
print(f"Parámetro de equilibrio entre velocidad/calidad de búsqueda: ef={p.ef}")


# Buscar en el índice HNSW
new_vector = get_embedding("una pelicula que trate sobre la busque de lugares escondidos debajo del agua")

labels, distances = p.knn_query(new_vector, k=10)

# asociar etiquetas con títulos de películas y mostrarlos
similar_movies = [(list(movies.keys())[label], round(1 - distance, 3)) for label, distance in zip(labels[0], distances[0])]
print(pd.DataFrame(similar_movies, columns=['película', 'similitud']))
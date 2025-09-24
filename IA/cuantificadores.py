import json

def scalar_quantization(embeddings: list[list[float]], num_levels=256, min_level=-128, max_level=127):
    # Aplanamos los embeddings para encontrar el mínimo y máximo global
    flattened_embeddings = [
        number for embedding in embeddings for number in embedding
    ]
    min_val = min(flattened_embeddings)
    max_val = max(flattened_embeddings)
    
    # Normalizamos los embeddings a [0, 1]
    normalized_embeddings = [
        [(number - min_val) / (max_val - min_val) for number in embedding]
        for embedding in embeddings
    ]
    
    # Cuantizamos los valores normalizados al número de niveles especificado
    quantized_embeddings = [
        [int(round(number * (num_levels - 1)) * (max_level - min_level) / (num_levels - 1) + min_level)
        for number in embedding]
        for embedding in normalized_embeddings
    ]
    
    return quantized_embeddings

def binary_quantization(embeddings):
    """Convertimos un float32 en un bit basado en el umbral proporcionado"""
    
    # encontramos la media de todas las dimensiones
    flattened_embeddings = [
        number for embedding in embeddings for number in embedding
    ]
    mean_val = sum(flattened_embeddings) / len(flattened_embeddings)
    
    # cuantizamos los embeddings a 1 bit
    quantized_embeddings = [
        [1 if number > mean_val else 0 for number in embedding]
        for embedding in embeddings
    ]
    return quantized_embeddings

with open('Vectores/peliculas_text-embedding-3-small-1536.json') as f:
    movies = json.load(f)

quantized_embeddings = scalar_quantization(list(movies.values()))
movies_1byte = {
    movie: quantized_embedding
    for movie, quantized_embedding in zip(movies.keys(), quantized_embeddings)
}

binary_quantized_embeddings = binary_quantization(list(movies.values()))
movies_1bit = {
    movie: quantized_embedding
    for movie, quantized_embedding in zip(movies.keys(), binary_quantized_embeddings)
}

print('Cuanttficacion:')
print('Flotante:')
print(movies['El Rey León'][0:10])
# Comprobamos los primeros 10 bytes del vector cuantizado para 'El Rey León'
print('Entero:')
print(movies_1byte['El Rey León'][0:10])
# Comprobamos los primeros 10 bits del vector cuantizado para 'El Rey León'
print('Binario:')
print(movies_1bit['El Rey León'][0:10])
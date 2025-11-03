import json
import pandas as pd


def cosine_similarity(v1, v2):

    dot_product = sum(
        [a * b for a, b in zip(v1, v2)])
    
    magnitude = (
        sum([a**2 for a in v1]) *
        sum([a**2 for a in v2])) ** 0.5

    return dot_product / magnitude

def most_similar(word: str, vectors: dict) -> list[list]:
    """Devuelve las 10 palabras más similares y sus similitudes respecto a la palabra dada"""
    word_vector = vectors[word]
    similarities = {w: cosine_similarity(word_vector, vector) for w, vector in vectors.items()}
    most_similar_words = sorted(similarities, key=similarities.get, reverse=True)
    return pd.DataFrame([(word, similarities[word]) for word in most_similar_words[1:10]], columns=['palabra', 'similitud'])

with open('Vectores/peliculas_text-embedding-3-small-1536.json') as f:
    movies_1536 = json.load(f)

with open('Vectores/peliculas_text-embedding-3-small-256.json') as f:
    movies_256t = json.load(f)

print(len(movies_1536['El Rey León']))
print(movies_1536['El Rey León'][0:4])

print(len(movies_256t['El Rey León']))
print(movies_256t['El Rey León'][0:4])

print(len(movies_1536['El Rey León']))

print(most_similar('El Rey León', movies_1536)[:10])

print(len(movies_256t['El Rey León']))

print(most_similar('El Rey León', movies_256t)[:10])
import json
import pandas as pd
    
with open('Vectores/sustantivos_text-embedding-ada-002.json') as f:
    vectors_ada2 = json.load(f)

with open('Vectores/sustantivos_text-embedding-ada-002.json') as f:
    vectors_emb3 = json.load(f)


def cosine_similarity(v1, v2):

    dot_product = sum(
        [a * b for a, b in zip(v1, v2)])
    
    magnitude = (
        sum([a**2 for a in v1]) *
        sum([a**2 for a in v2])) ** 0.5

    return dot_product / magnitude

vec1 = vectors_emb3['piel']
vec2 = vectors_emb3['carne']

print(cosine_similarity(vec1, vec2))


vec1 = vectors_emb3['bebe']
vec2 = vectors_emb3['adulto']

print(cosine_similarity(vec1, vec2))

def most_similar(word: str, vectors: dict) -> list[list]:
    """Devuelve las 10 palabras más similares y sus similitudes respecto a la palabra dada"""
    word_vector = vectors[word]
    similarities = {w: cosine_similarity(word_vector, vector) for w, vector in vectors.items()}
    most_similar_words = sorted(similarities, key=similarities.get, reverse=True)
    return pd.DataFrame([(word, similarities[word]) for word in most_similar_words[1:10]], columns=['palabra', 'similitud'])

word = 'bebe'
print(most_similar(word, vectors_ada2))
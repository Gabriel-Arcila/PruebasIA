import json
import random
import matplotlib.pyplot as plt
import pandas as pd


with open('Vectores/images_ai-vision.json') as f:
    image_vectors = json.load(f)

with open('Vectores/sustantivos_ai-vision.json') as f:
    word_vectors = json.load(f)


# random_image_name = random.choice(list(image_vectors.keys()))
# vector = image_vectors[random_image_name]

# plt.bar(range(len(vector)), vector)
# plt.xlabel('Dimension')
# plt.ylabel('Value')
# plt.show()


def cosine_similarity(v1, v2):
    """Calcula la similitud del coseno entre dos vectores"""
    dot_product = sum([a * b for a, b in zip(v1, v2)])
    magnitude = (sum([a**2 for a in v1]) * sum([a**2 for a in v2])) ** 0.5
    return dot_product / magnitude

def most_similar(target_vector: str, vectors: dict) -> list[list]:
    """Devuelve las imágenes más similares y sus similitudes respecto a las imágenes dadas"""
    similarities = {w: cosine_similarity(target_vector, vector) for w, vector in vectors.items()}
    most_similar = sorted(similarities, key=similarities.get, reverse=True)
    return pd.DataFrame([(vector_key, similarities[vector_key]) for vector_key in most_similar], columns=['vector key', 'similarity'])

# Renderiza la imagen objetivo
target_image = "inhaleexhale_top.jpg"
plt.imshow(plt.imread(f"img/{target_image}"))

most_similar_df = most_similar(image_vectors[target_image], image_vectors)[0:10]
print(most_similar_df)

# Ahora renderiza cada una de esas imágenes
# for image_name in most_similar_df['vector key'][1:]:
#     plt.imshow(plt.imread(f'../product_images/{image_name}'))
#     plt.axis('off')
#     plt.show()

target_word = "cama"
most_similar_df = most_similar(word_vectors[target_word], image_vectors)

print(f"\n{most_similar_df}")

# for image_name in most_similar_df['vector key'][0:10]:
#     plt.imshow(plt.imread(f'../product_images/{image_name}'))
#     plt.axis('off')
#     plt.show()
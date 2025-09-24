import numpy as np

vector1 = np.array([1, 2, 3])
vector2 = np.array([3, 1, 2])

def magnitude(vector):
    return sum([a**2 for a in vector]) ** 0.5

print('vectores:')
print(vector1)
print(vector2)

print('Magnitud de los vectores:')
print(np.linalg.norm(vector1))
print(magnitude(vector1)) # 3.7416573867739413
print(np.linalg.norm(vector2))
print(magnitude(vector2))


print('Normalizamos los vectores:')
v1_unit = vector1 / np.linalg.norm(vector1)
v2_unit = vector2 / np.linalg.norm(vector2)
print(v1_unit)
print(v2_unit)

mag_v1_unit = np.linalg.norm(v1_unit)
mag_v2_unit = np.linalg.norm(v2_unit)
print(mag_v1_unit)
print(mag_v2_unit)

v1 = v1_unit
v2 = v2_unit

print('Calculamos la distancia euclidiana (Es la distancia en línea recta entre dos puntos en el espacio euclidiano):')
distance = np.sqrt(np.sum((v1 - v2)**2))
print(distance) # o np.linalg.norm(v1 - v2)
distance = magnitude(v1 - v2)
print(distance)

print('Calculamos la distancia de Manhattan (Es la suma de las diferencias absolutas de sus coordenadas):')
distance = np.sum(np.abs(v1 - v2))
print(distance)

print('Calculamos el producto punto (es el producto escalar entre dos vectores):')
distance = sum(a * b for a, b in zip(v1, v2))
print(distance)

print('Calculamos la distancia del coseno (Es una medida de similitud entre dos vectores):')
similitud = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print('Similitud')
print(similitud) 
distance = 1 - similitud
print('Distancia')
print(distance)



print('Calculamos el producto punto (es el producto escalar entre dos vectores):')
distance = sum(a * b for a, b in zip(vector1, vector2))
print(distance)
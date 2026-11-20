# Proyecto: Clasificación de especies de flores Iris
# Algoritmo: K-Nearest Neighbors (KNN)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 1. Cargar el dataset Iris
iris = load_iris()

X = iris.data
y = iris.target

print("CLASIFICACIÓN DE ESPECIES DE FLORES IRIS")
print("=" * 50)

print("Cantidad de flores:", len(X))
print("Características:", iris.feature_names)
print("Especies:", iris.target_names)


# 2. Dividir los datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDIVISIÓN DE DATOS")
print("Datos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))


# 3. Crear el modelo KNN y escalar los datos
modelo = Pipeline([
    ("escalador", StandardScaler()),
    ("clasificador", KNeighborsClassifier(n_neighbors=5))
])


# 4. Entrenar el modelo
modelo.fit(X_train, y_train)

print("\nEl modelo se entrenó correctamente.")


# 5. Realizar predicciones
predicciones = modelo.predict(X_test)


# 6. Evaluar el modelo
exactitud = accuracy_score(y_test, predicciones)

print("\nRESULTADOS DE LA EVALUACIÓN")
print(f"Exactitud: {exactitud * 100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    predicciones,
    target_names=iris.target_names,
    zero_division=0
))


# 7. Mostrar la matriz de confusión
print("MATRIZ DE CONFUSIÓN")
print(confusion_matrix(y_test, predicciones))


# 8. Probar con una flor nueva
# Orden: largo y ancho del sépalo,
# largo y ancho del pétalo.

flor_nueva = [[6.4, 5.2, 5.2, 1.4]]

resultado = modelo.predict(flor_nueva)[0]

probabilidades = modelo.predict_proba(flor_nueva)[0]

print("\nCLASIFICACIÓN DE UNA FLOR NUEVA")
print("Medidas:", flor_nueva[0])

print("Especie identificada:", iris.target_names[resultado])

print("\nProbabilidades estimadas:")

for especie, probabilidad in zip(
    iris.target_names, probabilidades
):
    print(f"{especie}: {probabilidad * 100:.2f}%")
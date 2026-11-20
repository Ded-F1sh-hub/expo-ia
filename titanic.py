
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 1. Cargar el dataset
datos = pd.read_csv("train.csv")

print("PRIMEROS REGISTROS DEL DATASET")
print(datos.head())

# 2. Seleccionar las caracteristicas
caracteristicas = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = datos[caracteristicas]
y = datos["Survived"]

# 3. Limpiar datos faltantes
X = X.copy()

X["Age"] = X["Age"].fillna(X["Age"].median())
X["Fare"] = X["Fare"].fillna(X["Fare"].median())
X["Embarked"] = X["Embarked"].fillna(
    X["Embarked"].mode()[0]
)

# 4. Convertir datos de texto a numeros
X = pd.get_dummies(X, columns=["Sex", "Embarked"])

# 5. Separar datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Crear el modelo
# Crear bosque aleatorio de 100 árboles
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

# 7. Entrenar el modelo
modelo.fit(X_train, y_train)

print("\nMODELO ENTRENADO CORRECTAMENTE")

# 8. Realizar predicciones
predicciones = modelo.predict(X_test)

# 9. Evaluar el modelo
precision = accuracy_score(y_test, predicciones)

print("\nPRECISION DEL MODELO:")
print(f"{precision * 100:.2f}%")

print("\nREPORTE DE CLASIFICACION:")
print(classification_report(y_test, predicciones))

print("\nMATRIZ DE CONFUSION:")
print(confusion_matrix(y_test, predicciones))


# 10. Predecir la supervivencia de un pasajero
print("\n--- PREDICCION DE SUPERVIVENCIA ---")

edad = float(input("Ingresa la edad: "))
sexo = input("Sexo (male/female): ").lower()
clase = int(input("Clase del pasajero (1, 2 o 3): "))
hermanos = int(input("Numero de hermanos o pareja a bordo: "))
padres = int(input("Numero de padres o hijos a bordo: "))
tarifa = float(input("Tarifa del boleto: "))
embarque = input("Puerto (C, Q o S): ").upper()

# Crear registro del nuevo pasajero
nuevo = pd.DataFrame([{
    "Pclass": clase,
    "Sex": sexo,
    "Age": edad,
    "SibSp": hermanos,
    "Parch": padres,
    "Fare": tarifa,
    "Embarked": embarque
}])

# Aplicar la misma conversion de variables
nuevo = pd.get_dummies(
    nuevo,
    columns=["Sex", "Embarked"]
)

# Asegurar que tenga las mismas columnas del entrenamiento
nuevo = nuevo.reindex(
    columns=X.columns,
    fill_value=0
)

# Realizar prediccion
resultado = modelo.predict(nuevo)

print("\nRESULTADO DE LA PREDICCION:")

if resultado[0] == 1:
    print("El pasajero SOBREVIVIO (prediccion).")
else:
    print("El pasajero NO SOBREVIVIO (prediccion).")

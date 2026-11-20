
import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw, ImageOps
from PIL import ImageFilter

import os

# ==========================================
# 1. CONFIGURACION
# ==========================================

MODELO = "modelo_mnist.keras"

# ==========================================
# 2. ENTRENAMIENTO DE LA RED NEURONAL
# ==========================================

def entrenar_modelo():

    print("Cargando dataset MNIST...")

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalizar los pixeles
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Agregar canal de color
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("Creando red neuronal CNN...")

    modelo = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128, activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            10, activation="softmax"
        )
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Entrenando modelo...")

    modelo.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1
    )

    perdida, precision = modelo.evaluate(
        x_test, y_test, verbose=0
    )

    print("Precision:", precision * 100, "%")

    modelo.save(MODELO)

    print("Modelo guardado correctamente.")

    return modelo


# ==========================================
# 3. CARGAR O ENTRENAR MODELO
# ==========================================

if os.path.exists(MODELO):

    print("Cargando modelo existente...")
    modelo = tf.keras.models.load_model(MODELO)

else:

    modelo = entrenar_modelo()


# ==========================================
# 4. INTERFAZ GRAFICA
# ==========================================

VENTANA = 280

ventana = tk.Tk()
ventana.title("Reconocimiento de Digitos MNIST")
ventana.geometry("450x570")
ventana.resizable(False, False)
ventana.configure(bg="#20232a")

# Titulo
titulo = tk.Label(
    ventana,
    text="RECONOCIMIENTO MNIST",
    font=("Arial", 17, "bold"),
    bg="#20232a",
    fg="white"
)

titulo.pack(pady=15)

subtitulo = tk.Label(
    ventana,
    text="Dibuja un numero del 0 al 9",
    font=("Arial", 12),
    bg="#20232a",
    fg="white"
)

subtitulo.pack(pady=5)


# ==========================================
# 5. LIENZO PARA DIBUJAR
# ==========================================

lienzo = tk.Canvas(
    ventana,
    width=VENTANA,
    height=VENTANA,
    bg="black",
    cursor="cross"
)

lienzo.pack(pady=15)

# Imagen interna para procesar
imagen = Image.new("L", (VENTANA, VENTANA), 0)

dibujo = ImageDraw.Draw(imagen)


# ==========================================
# 6. FUNCIONES DE DIBUJO
# ==========================================

def pintar(event):

    x = event.x
    y = event.y

    radio = 10

    # Dibujar en pantalla
    lienzo.create_oval(
        x-radio,
        y-radio,
        x+radio,
        y+radio,
        fill="white",
        outline="white"
    )

    # Dibujar en imagen
    dibujo.ellipse(
        (x-radio, y-radio, x+radio, y+radio),
        fill=255
    )


lienzo.bind("<B1-Motion>", pintar)


# ==========================================
# 7. PROCESAMIENTO DE LA IMAGEN
# ==========================================

def procesar_dibujo():

    img = imagen.copy()

    # Suavizar un poco los trazos
    img = img.filter(ImageFilter.GaussianBlur(1))

    # Encontrar el area del numero
    bbox = img.getbbox()

    if bbox is None:
        return None

    img = img.crop(bbox)

    # Ajustar a 20x20 conservando proporciones
    img.thumbnail((20, 20))

    # Centrar en una imagen de 28x28
    nueva = Image.new("L", (28, 28), 0)

    x = (28 - img.width) // 2
    y = (28 - img.height) // 2

    nueva.paste(img, (x, y))

    # Normalizar
    datos = np.array(nueva).astype("float32") / 255.0

    # Preparar entrada para CNN
    datos = datos.reshape(1, 28, 28, 1)

    return datos


# ==========================================
# 8. RECONOCIMIENTO DEL DIGITO
# ==========================================

def reconocer():

    datos = procesar_dibujo()

    if datos is None:
        messagebox.showwarning(
            "Aviso",
            "Primero debes dibujar un numero."
        )
        return

    predicciones = modelo.predict(datos, verbose=0)[0]

    numero = np.argmax(predicciones)

    confianza = predicciones[numero] * 100

    resultado.config(
        text=f"Numero reconocido: {numero}",
        fg="#00ff88"
    )

    porcentaje.config(
        text=f"Confianza: {confianza:.2f}%"
    )

    # Mostrar probabilidades de los 10 digitos
    probabilidades.config(
        text="\n".join(
            [f"{i}: {predicciones[i]*100:.2f}%"
             for i in range(10)]
        )
    )


# ==========================================
# 9. LIMPIAR LIENZO
# ==========================================

def limpiar():

    lienzo.delete("all")

    global imagen, dibujo

    imagen = Image.new("L", (VENTANA, VENTANA), 0)

    dibujo = ImageDraw.Draw(imagen)

    resultado.config(
        text="Numero reconocido: --",
        fg="white"
    )

    porcentaje.config(text="Confianza: --")

    probabilidades.config(text="")


# ==========================================
# 10. BOTONES
# ==========================================

botones = tk.Frame(ventana, bg="#20232a")
botones.pack(pady=10)

btn_reconocer = tk.Button(
    botones,
    text="RECONOCER",
    command=reconocer,
    bg="#007bff",
    fg="white",
    font=("Arial", 11, "bold"),
    width=14
)

btn_reconocer.grid(row=0, column=0, padx=8)

btn_limpiar = tk.Button(
    botones,
    text="LIMPIAR",
    command=limpiar,
    bg="#dc3545",
    fg="white",
    font=("Arial", 11, "bold"),
    width=14
)

btn_limpiar.grid(row=0, column=1, padx=8)


# ==========================================
# 11. RESULTADOS
# ==========================================

resultado = tk.Label(
    ventana,
    text="Numero reconocido: --",
    font=("Arial", 15, "bold"),
    bg="#20232a",
    fg="white"
)

resultado.pack(pady=8)

porcentaje = tk.Label(
    ventana,
    text="Confianza: --",
    font=("Arial", 12),
    bg="#20232a",
    fg="white"
)

porcentaje.pack()

probabilidades = tk.Label(
    ventana,
    text="",
    font=("Consolas", 10),
    bg="#20232a",
    fg="#cccccc",
    justify="left"
)

probabilidades.pack(pady=5)

# ==========================================
# 12. INICIAR PROGRAMA
# ==========================================

ventana.mainloop()

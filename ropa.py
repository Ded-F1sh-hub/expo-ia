import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Cargar dataset Fashion-MNIST
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

clases = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# --- VISUALIZACIÓN 1: Muestras originales del dataset ---
plt.figure(figsize=(10, 4))
for i in range(10):
  plt.subplot(2, 5, i + 1)
  plt.imshow(x_train[i], cmap="gray")
  plt.title(clases[y_train[i]], fontsize=9)
  plt.axis("off")
plt.suptitle("Ejemplos del Dataset Fashion-MNIST")
plt.tight_layout()
plt.show()

# 2. Preprocesamiento (Normalización y dimensión de canal)
x_train = (x_train / 255.0)[..., np.newaxis]
x_test = (x_test / 255.0)[..., np.newaxis]

# 3. Construcción y entrenamiento del modelo (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train, epochs=10, batch_size=128, validation_split=0.1
)

# --- VISUALIZACIÓN 2: Curvas de entrenamiento ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(history.history["accuracy"], label="Entrenamiento")
ax1.plot(history.history["val_accuracy"], label="Validación")
ax1.set_title("Exactitud (Accuracy) por Época")
ax1.set_xlabel("Época")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(history.history["loss"], label="Entrenamiento")
ax2.plot(history.history["val_loss"], label="Validación")
ax2.set_title("Pérdida (Loss) por Época")
ax2.set_xlabel("Época")
ax2.set_ylabel("Loss")
ax2.legend()
plt.tight_layout()
plt.show()

# 4. Evaluación con datos de prueba
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nExactitud final en prueba: {test_acc*100:.2f}%\n")

y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=clases))

# --- VISUALIZACIÓN 3: Matriz de Confusión ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(9, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, colorbar=False)
plt.title("Matriz de Confusión - Fashion-MNIST")
plt.tight_layout()
plt.show()

# --- VISUALIZACIÓN 4: Galería de los 15 ejemplos en prueba ---
plt.figure(figsize=(11, 6))
for i in range(15):
  plt.subplot(3, 5, i + 1)
  plt.imshow(x_test[i].squeeze(), cmap="gray")
  real = clases[y_test[i]]
  pred = clases[y_pred[i]]
  conf = np.max(y_pred_probs[i]) * 100
  color = "green" if real == pred else "red"
  plt.title(f"Pred: {pred} ({conf:.0f}%)\nReal: {real}", color=color, fontsize=8)
  plt.axis("off")
plt.suptitle("Predicciones sobre el Conjunto de Prueba (x_test)")
plt.tight_layout()
plt.show()


# --- VISUALIZACIÓN 5: Interfaz para dibujar y probar muestra aleatoria ---
def abrir_interfaz_interactiva():
  canvas_size = 280
  img_pil = Image.new("L", (canvas_size, canvas_size), color=0)
  draw = ImageDraw.Draw(img_pil)
  ultimo_punto = [None, None]
  photo_ref = [None]

  ventana = tk.Tk()
  ventana.title("Dibujar o Probar Muestra Aleatoria - Fashion-MNIST")

  canvas = tk.Canvas(
      ventana,
      width=canvas_size,
      height=canvas_size,
      bg="black",
      highlightthickness=1,
      highlightbackground="#666666",
  )
  canvas.pack(padx=12, pady=(12, 6))

  lbl_info = tk.Label(
      ventana,
      text="Dibuja una prenda o presiona 'Muestra aleatoria'",
      font=("Arial", 10),
      justify="center",
  )
  lbl_info.pack(pady=6)

  def predecir_desde_pil(pil_img, etiqueta_real=None):
    img_28 = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
    arr = (np.array(img_28, dtype=np.float32) / 255.0).reshape(1, 28, 28, 1)
    probs = model.predict(arr, verbose=0)[0]
    idx_pred = int(np.argmax(probs))
    conf = probs[idx_pred] * 100.0

    texto = f"Predicción: {clases[idx_pred]} ({conf:.1f}%)"
    if etiqueta_real is not None:
      texto += f"\nEtiqueta real: {clases[etiqueta_real]}"
    lbl_info.config(text=texto)

  def limpiar():
    nonlocal img_pil, draw
    canvas.delete("all")
    img_pil = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img_pil)
    ultimo_punto[:] = [None, None]
    photo_ref[0] = None
    lbl_info.config(text="Lienzo limpio. Dibuja o elige una muestra aleatoria.")

  def al_presionar(event):
    ultimo_punto[:] = [event.x, event.y]
    r = 11
    canvas.create_oval(
        event.x - r,
        event.y - r,
        event.x + r,
        event.y + r,
        fill="white",
        outline="white",
    )
    draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)

  def al_arrastrar(event):
    if ultimo_punto[0] is not None and ultimo_punto[1] is not None:
      x0, y0 = ultimo_punto[0], ultimo_punto[1]
      x1, y1 = event.x, event.y
      canvas.create_line(
          x0, y0, x1, y1, fill="white", width=22, capstyle=tk.ROUND, smooth=True
      )
      draw.line([x0, y0, x1, y1], fill=255, width=22)
    ultimo_punto[:] = [event.x, event.y]

  def al_soltar(_event):
    ultimo_punto[:] = [None, None]

  def muestra_aleatoria():
    nonlocal img_pil, draw
    idx = int(np.random.randint(0, len(x_test)))
    arr_28 = (x_test[idx].squeeze() * 255.0).astype(np.uint8)
    pil_28 = Image.fromarray(arr_28, mode="L")
    img_grande = pil_28.resize(
        (canvas_size, canvas_size), Image.Resampling.NEAREST
    )

    img_pil = img_grande.copy()
    draw = ImageDraw.Draw(img_pil)

    canvas.delete("all")
    tk_img = ImageTk.PhotoImage(img_grande)
    photo_ref[0] = tk_img
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    predecir_desde_pil(pil_28, etiqueta_real=int(y_test[idx]))

  canvas.bind("<ButtonPress-1>", al_presionar)
  canvas.bind("<B1-Motion>", al_arrastrar)
  canvas.bind("<ButtonRelease-1>", al_soltar)

  frame_btns = tk.Frame(ventana)
  frame_btns.pack(pady=(4, 12))

  tk.Button(
      frame_btns, text="Clasificar dibujo", command=lambda: predecir_desde_pil(img_pil)
  ).grid(row=0, column=0, padx=5)
  tk.Button(frame_btns, text="Muestra aleatoria", command=muestra_aleatoria).grid(
      row=0, column=1, padx=5
  )
  tk.Button(frame_btns, text="Limpiar", command=limpiar).grid(
      row=0, column=2, padx=5
  )

  ventana.mainloop()


abrir_interfaz_interactiva()
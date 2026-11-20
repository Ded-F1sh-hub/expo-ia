import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CARGA DE DATOS, ESCALADO Y ENTRENAMIENTO
# ==========================================
def load_and_train():
    # Cargar datos originales
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Selección de características (evitamos repetir variables de tamaño)
    features = ['mean radius', 'mean texture', 'mean smoothness', 'mean concavity', 'mean symmetry']
    X = df[features]
    y = df['target']

    # Dividir los datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Estandarizar (Ecualizador matemático)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento del modelo con datos estandarizados
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_scaled, y_train)

    # Calcular precisión
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, X, features, accuracy

# Ejecutamos la carga y entrenamiento una vez al abrir el programa
model, scaler, X, features, accuracy = load_and_train()

# ==========================================
# 2. DISEÑO DE LA INTERFAZ GRÁFICA (Tkinter)
# ==========================================

# Crear la ventana principal
root = tk.Tk()
root.title("🩺 Sistema de Diagnóstico Clínico")
root.geometry("850x550")
root.configure(padx=20, pady=20)

# Estilos básicos
style = ttk.Style()
style.theme_use('clam')
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", font=("Arial", 10))
style.configure("Header.TLabel", font=("Arial", 16, "bold"))
style.configure("Metric.TLabel", font=("Arial", 10, "bold"), foreground="#0052cc")

# Encabezado principal
ttk.Label(root, text="🩺 Panel de Apoyo al Diagnóstico Médico", style="Header.TLabel").pack(pady=(0, 5))
ttk.Label(root, text="Interfaz interactiva basada en el dataset Breast Cancer Wisconsin.").pack(pady=(0, 15))

# Fila de métricas
metrics_frame = ttk.Frame(root)
metrics_frame.pack(fill="x", pady=10)
ttk.Label(metrics_frame, text=f"📊 Precisión del Modelo: {accuracy * 100:.2f}%", style="Metric.TLabel").pack(side="left", padx=20)
ttk.Label(metrics_frame, text=f"🔍 Variables Analizadas: {len(features)}", style="Metric.TLabel").pack(side="left", padx=20)
ttk.Label(metrics_frame, text="⚙️ Algoritmo: Regresión Logística", style="Metric.TLabel").pack(side="left", padx=20)

ttk.Separator(root, orient="horizontal").pack(fill="x", pady=15)

# Contenedor principal dividido en dos "columnas" (Izquierda y Derecha)
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_col = tk.LabelFrame(main_frame, text="🎛️ Parámetros de la Muestra", font=("Arial", 11, "bold"), padx=15, pady=15)
left_col.pack(side="left", fill="y", padx=(0, 20))

right_col = tk.Frame(main_frame)
right_col.pack(side="left", fill="both", expand=True)

# Crear los controles deslizantes (Sliders) dinámicamente según el dataset
sliders = {}
for feature in features:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    
    label_name = feature.replace("mean ", "").capitalize()
    ttk.Label(left_col, text=label_name).pack(anchor="w", pady=(10, 0))
    
    var = tk.DoubleVar(value=mean_val)
    # tk.Scale muestra el número dinámicamente al lado
    slider = tk.Scale(left_col, variable=var, from_=min_val, to=max_val, resolution=0.0001, 
                      orient="horizontal", length=250, borderwidth=0, highlightthickness=0)
    slider.pack(anchor="w")
    sliders[feature] = var

# Función para realizar la predicción cuando se presiona el botón
def ejecutar_diagnostico():
    # 1. Recolectar valores actuales de los sliders
    user_inputs = {feat: [sliders[feat].get()] for feat in features}
    input_df = pd.DataFrame(user_inputs)
    
    # 2. Pasar los datos por el ecualizador matemático (StandardScaler)
    input_scaled = scaler.transform(input_df)
    
    # 3. Realizar predicción
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # 0 = Maligno, 1 = Benigno (según la codificación de scikit-learn)
    if prediction == 1:
        confidence = probabilities[1] * 100
        resultado_lbl.config(
            text=f"✅ Diagnóstico: BENIGNO\nConfianza del modelo: {confidence:.2f}%\n\nLa muestra presenta características\nasociadas a tejido benigno.",
            fg="green"
        )
    else:
        confidence = probabilities[0] * 100
        resultado_lbl.config(
            text=f"⚠️ Diagnóstico: MALIGNO\nConfianza del modelo: {confidence:.2f}%\n\nLa muestra presenta características\nasociadas a tejido maligno.\nSe sugiere revisión especializada.",
            fg="#d90000"
        )

# Botón y área de resultados en la columna derecha
ttk.Label(right_col, text="📊 Resultado del Análisis", font=("Arial", 12, "bold")).pack(pady=(10, 20))

btn_diagnosticar = tk.Button(right_col, text="🔍 Ejecutar Diagnóstico Clínico", 
                             command=ejecutar_diagnostico, 
                             bg="#0052cc", fg="white", font=("Arial", 12, "bold"), 
                             padx=20, pady=10, relief="flat", cursor="hand2")
btn_diagnosticar.pack(pady=10)

# Etiqueta donde se mostrará el resultado
resultado_lbl = tk.Label(right_col, text="👈 Ajusta los valores en el panel izquierdo\ny presiona 'Ejecutar Diagnóstico'.", 
                         font=("Arial", 12), justify="center", fg="gray", height=6)
resultado_lbl.pack(pady=30)

# Iniciar la aplicación
root.mainloop()
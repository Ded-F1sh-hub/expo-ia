import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==========================================
# 1. CARGA DE DATOS, ESCALADO Y ENTRENAMIENTO
# ==========================================
def load_and_train():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    
    # El precio original viene expresado en cientos de miles de dólares
    # Lo multiplicamos por 100,000 para trabajar con dólares reales
    df['Price'] = california.target * 100000 

    features = ['MedInc', 'HouseAge', 'AveRooms', 'Population']
    X = df[features]
    y = df['Price']

    # Dividir: 80% para entrenar, 20% para examen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ecualizador matemático (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluar precisión (En regresión se usa R² en lugar de Accuracy)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, df, features, r2

model, scaler, df, features, r2 = load_and_train()

# ==========================================
# 2. DISEÑO DE LA INTERFAZ GRÁFICA (Dark Mode)
# ==========================================
root = tk.Tk()
root.title("🏢 IA Inmobiliaria - Tasador Inteligente")
root.geometry("550x650")

# Paleta de colores elegante
BG_MAIN = "#121212"    
BG_CARD = "#1E1E1E"    
FG_TEXT = "#E0E0E0"    
ACCENT = "#D4AF37"     

root.configure(bg=BG_MAIN)

# --- Encabezados ---
titulo = tk.Label(root, text="🏢 Tasador Inmobiliario IA", font=("Helvetica", 22, "bold"), bg=BG_MAIN, fg=ACCENT)
titulo.pack(pady=(25, 5))

subtitulo = tk.Label(root, text="Calcula el valor de la propiedad ingresando los datos exactos.", font=("Helvetica", 10), bg=BG_MAIN, fg="#AAAAAA")
subtitulo.pack(pady=(0, 20))

# --- Contenedor Central (Tarjeta 2x2) ---
card_frame = tk.Frame(root, bg=BG_CARD, padx=25, pady=25, highlightbackground="#333333", highlightthickness=1)
card_frame.pack(fill="x", padx=40)

# Diccionario con valores por defecto y ajustes iniciales
campos_info = {
    'MedInc': {'label': 'Ingreso Promedio ($):', 'default': round(df['MedInc'].median() * 10000, 2)},
    'HouseAge': {'label': 'Antigüedad (Años):', 'default': round(df['HouseAge'].median(), 1)},
    'AveRooms': {'label': 'Promedio Habitaciones:', 'default': round(df['AveRooms'].median(), 1)},
    'Population': {'label': 'Población de la Zona:', 'default': int(df['Population'].median())}
}

entradas = {}

# Crear un grid 2x2 dinámico
for i, feature in enumerate(features):
    row = i // 2 
    col = i % 2  
    
    frame_input = tk.Frame(card_frame, bg=BG_CARD)
    frame_input.grid(row=row, column=col, padx=15, pady=15, sticky="ew")
    
    lbl = tk.Label(frame_input, text=campos_info[feature]['label'], font=("Helvetica", 9, "bold"), bg=BG_CARD, fg=FG_TEXT)
    lbl.pack(anchor="w")
    
    ent = tk.Entry(frame_input, font=("Helvetica", 13), width=15, bg="#2C2C2C", fg="#FFFFFF", insertbackground="#FFFFFF", relief="flat")
    ent.insert(0, str(campos_info[feature]['default']))
    ent.pack(anchor="w", pady=5)
    
    entradas[feature] = ent

# --- Función de predicción con restricciones de tipo y lógica ---
def ejecutar_prediccion():
    try:
        # 1. Parseo mixto: float para decimales, int estricto para población
        ingreso_real = float(entradas['MedInc'].get())
        habitaciones = float(entradas['AveRooms'].get())
        antiguedad = float(entradas['HouseAge'].get()) 
        poblacion = int(entradas['Population'].get())  
        
        # 2. Restricciones Lógicas (Control de errores biológicos/físicos)
        if poblacion <= 0 or antiguedad < 0 or habitaciones <= 0 or ingreso_real < 0:
            messagebox.showwarning("Datos Inválidos", "No puedes ingresar valores negativos ni una población de cero.")
            return

        # 3. Formatear ingreso para la estructura del dataset
        ingreso_modelo = ingreso_real / 10000.0 
        
        user_inputs = {
            'MedInc': [ingreso_modelo],
            'HouseAge': [antiguedad],
            'AveRooms': [habitaciones],
            'Population': [poblacion]
        }
        
        input_df = pd.DataFrame(user_inputs)
        
        # 4. Aplicar escalador y modelo matemático
        input_scaled = scaler.transform(input_df)
        predicted_price = model.predict(input_scaled)[0]
        
        # Prevenir resultados financieros negativos
        if predicted_price < 0:
            predicted_price = 0
            
        display_resultado.config(text=f"${predicted_price:,.2f} USD")
        
    except ValueError:
        messagebox.showerror(
            "Error de Formato", 
            "Por favor, revisa tus entradas:\n\n"
            "- Ingreso, Antigüedad (ej. 5.5) y Habitaciones aceptan decimales.\n"
            "- Población DEBE ser un número entero sin puntos."
        )

# --- Botón Principal ---
btn_predecir = tk.Button(root, text="💰 CALCULAR PRECIO ESTIMADO", command=ejecutar_prediccion,
                         bg=ACCENT, fg="#000000", font=("Helvetica", 13, "bold"), 
                         relief="flat", cursor="hand2", pady=12)
btn_predecir.pack(fill="x", padx=40, pady=25)

# --- Pantalla de Resultado (LED-Style) ---
display_frame = tk.Frame(root, bg="#000000", pady=20, highlightbackground="#333333", highlightthickness=2)
display_frame.pack(fill="x", padx=40)

lbl_res = tk.Label(display_frame, text="VALOR ESTIMADO EN EL MERCADO", font=("Helvetica", 9), bg="#000000", fg="#888888")
lbl_res.pack()

display_resultado = tk.Label(display_frame, text="$0.00 USD", font=("Courier", 26, "bold"), bg="#000000", fg="#00FF41")
display_resultado.pack(pady=5)

# --- Pie de página informativo ---
lbl_footer = tk.Label(root, text=f"📊 Precisión Matemática (R²): {r2 * 100:.1f}%", bg=BG_MAIN, fg="#666666", font=("Helvetica", 9))
lbl_footer.pack(side="bottom", pady=15)

root.mainloop()
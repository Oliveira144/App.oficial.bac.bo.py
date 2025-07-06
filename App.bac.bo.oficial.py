# --- Sistema de Backtesting Automático ---
st.markdown("---")
st.header("🧪 TESTE DE ESTRATÉGIAS")

# Função para estratégia preditora
def estrategia_preditor(df):
    if len(df) > 15:
        X_train = df[["Player", "Banker"]].values[:-1]
        y_train = df["Resultado"].values[1:]
        X_pred = df[["Player", "Banker"]].values[-1].reshape(1, -1)
        
        probas = previsao_avancada(X_train, y_train, X_pred)
        return ["P", "B", "T"][np.argmax(probas)]
    return "B"

# Estratégias para teste
estrategias = [
    {"nome": "Tendência Player", 
     "funcao": lambda df: "P" if df["Player"].mean() > df["Banker"].mean() else "B",
     "desc": "Aposta no lado com maior soma média"},
    
    {"nome": "Anti-Streak", 
     "funcao": lambda df: "P" if df["Resultado"].iloc[-1] == "B" else "B",
     "desc": "Aposta contra o último resultado"},
    
    {"nome": "Soma Alta Player", 
     "funcao": lambda df: "P" if df["Player"].iloc[-1] > 8 else "B",
     "desc": "Aposta no Player quando sua soma > 8"},
    
    {"nome": "Sistema Preditor", 
     "funcao": estrategia_preditor,
     "desc": "Usa o modelo de machine learning para prever"}
]

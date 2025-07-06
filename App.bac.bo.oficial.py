import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from datetime import datetime
import time
import io

# --- Configuração Premium da Página ---
st.set_page_config(
    page_title="Bac Bo Intelligence Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎯"
)
st.title("🎯 BAC BO PREDICTOR PRO - Sistema de Alta Precisão")

# Estilos CSS Premium (Mantidos sem alterações significativas, pois já estão bem definidos)
st.markdown("""
<style>
    /* Design Premium */
    .stApp {
        background: linear-gradient(135deg, #1a1b28, #26273b);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stAlert {
        padding: 1.8rem;
        border-radius: 15px;
        margin-bottom: 1.8rem;
        font-size: 1.4em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        border: 2px solid;
    }
    .alert-success {
        background: linear-gradient(135deg, #28a745, #1e7e34);
        border-color: #0c5420;
    }
    .alert-danger {
        background: linear-gradient(135deg, #dc3545, #bd2130);
        border-color: #8a1621;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffc107, #e0a800);
        border-color: #b38700;
        color: #000 !important;
    }
    .stMetric {
        background: rgba(46, 47, 58, 0.7);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border: 1px solid #3d4050;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Botões premium */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    /* Abas */
    .stTabs [aria-selected="true"] {
        font-weight: bold;
        background: rgba(46, 47, 58, 0.9) !important;
    }
    .css-1aumxhk { /* Pode precisar de ajuste dependendo da versão do Streamlit */
        background-color: rgba(38, 39, 48, 0.8) !important;
    }
    /* Melhorias na tabela */
    .dataframe th {
        background-color: #2a2b3c !important;
        color: white !important;
    }
    .dataframe tr:nth-child(even) {
        background-color: #2a2b3c !important;
    }
    .dataframe tr:nth-child(odd) {
        background-color: #1e1f2c !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Inicialização do Session State ---
# Usar um dicionário para 'backtest_results' e 'historico_recomendacoes' para evitar KeyError
if 'historico_dados' not in st.session_state:
    st.session_state.historico_dados = []
    st.session_state.padroes_detectados = []
    st.session_state.modelos_treinados = False
    st.session_state.ultimo_treinamento = None
    st.session_state.backtest_results = {} # Inicializado como dicionário vazio
    st.session_state.estrategia_atual = "Simples"
    st.session_state.historico_recomendacoes = [] # Inicializado como lista vazia

# --- Constantes Avançadas ---
JANELAS_ANALISE = [
    {"nome": "Ultra-curto", "tamanho": 8, "peso": 1.5},
    {"nome": "Curto", "tamanho": 20, "peso": 1.8},
    {"nome": "Médio", "tamanho": 50, "peso": 1.2},
    {"nome": "Longo", "tamanho": 100, "peso": 0.9}
]

MODELOS = {
    "XGBoost": xgb.XGBClassifier(n_estimators=150, learning_rate=0.12, max_depth=5, use_label_encoder=False, eval_metric='mlogloss'), # Adicionado eval_metric para remover warning
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42), # Adicionado random_state
    "Neural Network": MLPClassifier(hidden_layer_sizes=(25, 15), activation='relu', max_iter=2000, random_state=42), # Adicionado random_state
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # Adicionado random_state
}

# --- Funções de Análise Avançada ---

@st.cache_data(ttl=3600) # Cache para resultados da função por 1 hora
def calcular_probabilidade_condicional(df, evento_str, condicao_str):
    try:
        # Usar eval() para avaliar as strings de consulta
        df_condicao = df.query(condicao_str)
        total_condicao = len(df_condicao)
        if total_condicao == 0:
            return 0.0
        
        df_ambos = df_condicao.query(evento_str)
        total_ambos = len(df_ambos)
        return (total_ambos / total_condicao) * 100
    except Exception as e:
        st.error(f"Erro ao calcular probabilidade condicional: {e}")
        return 0.0

@st.cache_resource # Cache para modelos treinados
def previsao_avancada(X_train, y_train, X_pred):
    probas = []
    
    # Padronizar os dados de entrada
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    for nome, modelo in MODELOS.items():
        try:
            # Garante que o modelo é clonado para evitar treinamento em cima do mesmo objeto
            # para cada iteração do streamlit, que pode não ser stateless
            m = MODELOS[nome].__class__(**MODELOS[nome].get_params())
            m.fit(X_train_scaled, y_train)
            proba = m.predict_proba(X_pred_scaled)[0]
            probas.append(proba)
        except Exception as e:
            st.warning(f"Erro no modelo {nome}: {str(e)}. Pulando este modelo.") # Warning em vez de error para não parar a execução

    if probas:
        # Média ponderada ou simples das probabilidades
        return np.mean(probas, axis=0)
    return np.array([0.33, 0.33, 0.34])  # Retorno neutro se falhar

@st.cache_data(ttl=600) # Cache para padrões detectados
def detectar_padroes_avancados(df_completo):
    if df_completo.empty:
        return []

    todos_padroes = []
    
    # Certificar-se de que as colunas 'Player', 'Banker' e 'Resultado' existem
    if not all(col in df_completo.columns for col in ["Player", "Banker", "Resultado"]):
        st.error("DataFrame de entrada não contém as colunas necessárias (Player, Banker, Resultado).")
        return []

    for janela in JANELAS_ANALISE:
        tamanho = janela["tamanho"]
        peso_janela = janela["peso"]
        
        if len(df_completo) < tamanho:
            continue
            
        df_analise = df_completo.tail(tamanho).copy()
        n = len(df_analise)
        
        if n < 2: # Necessário pelo menos 2 pontos para regressão linear
            continue

        x = np.arange(n)
        
        # 1. Análise de Tendência Avançada
        try:
            # Usar iloc para evitar problemas de índice com copy()
            player_data = df_analise["Player"].values
            banker_data = df_analise["Banker"].values

            if len(x) > 1 and len(player_data) > 1: # Checar se há dados suficientes para regressão
                player_slope, _, _, _, _ = stats.linregress(x, player_data)
                player_trend_strength = min(2.5, abs(player_slope) * 8)
                
                if player_slope > 0.15:
                    todos_padroes.append({
                        "tipo": "TENDÊNCIA",
                        "lado": "P",
                        "desc": f"Soma Player em alta forte ({player_slope:.2f}) - Janela {janela['nome']}",
                        "peso": player_trend_strength * peso_janela,
                        "janela": janela["nome"]
                    })
                elif player_slope < -0.15:
                    todos_padroes.append({
                        "tipo": "TENDÊNCIA",
                        "lado": "P",
                        "desc": f"Soma Player em queda forte ({player_slope:.2f}) - Janela {janela['nome']}",
                        "peso": player_trend_strength * peso_janela,
                        "janela": janela["nome"]
                    })

            if len(x) > 1 and len(banker_data) > 1:
                banker_slope, _, _, _, _ = stats.linregress(x, banker_data)
                banker_trend_strength = min(2.5, abs(banker_slope) * 8)
                
                if banker_slope > 0.15:
                    todos_padroes.append({
                        "tipo": "TENDÊNCIA",
                        "lado": "B",
                        "desc": f"Soma Banker em alta forte ({banker_slope:.2f}) - Janela {janela['nome']}",
                        "peso": banker_trend_strength * peso_janela,
                        "janela": janela["nome"]
                    })
                elif banker_slope < -0.15:
                    todos_padroes.append({
                        "tipo": "TENDÊNCIA",
                        "lado": "B",
                        "desc": f"Soma Banker em queda forte ({banker_slope:.2f}) - Janela {janela['nome']}",
                        "peso": banker_trend_strength * peso_janela,
                        "janela": janela["nome"]
                    })
        except Exception as e:
            st.warning(f"Erro na análise de tendência para janela {janela['nome']}: {e}")
            pass # Continua se houver erro na regressão

        # 2. Análise de Repetição Estatística
        player_counts = Counter(df_analise["Player"])
        banker_counts = Counter(df_analise["Banker"])
        
        for soma, count in player_counts.items():
            if count >= max(4, n*0.35):  # Limiares mais rigorosos
                peso = min(3.0, count * 0.6) * peso_janela
                todos_padroes.append({
                    "tipo": "REPETIÇÃO",
                    "lado": "P",
                    "desc": f"Soma Player {soma} repetida {count}/{n} vezes ({count/n*100:.1f}%)",
                    "peso": peso,
                    "janela": janela["nome"]
                })
                
        for soma, count in banker_counts.items():
            if count >= max(4, n*0.35):
                peso = min(3.0, count * 0.6) * peso_janela
                todos_padroes.append({
                    "tipo": "REPETIÇÃO",
                    "lado": "B",
                    "desc": f"Soma Banker {soma} repetida {count}/{n} vezes ({count/n*100:.1f}%)",
                    "peso": peso,
                    "janela": janela["nome"]
                })
        
        # 3. Previsão com Modelo Híbrido
        if n > 15: # Mínimo de dados para treino e teste
            try:
                # Usar .values para garantir que são arrays NumPy
                X = df_analise[["Player", "Banker"]].values[:-1]
                y = df_analise["Resultado"].values[1:]
                X_pred = df_analise[["Player", "Banker"]].values[-1].reshape(1, -1)
                
                # Certificar-se de que X e y têm o mesmo número de amostras
                if len(X) == 0 or len(y) == 0 or len(X) != len(y):
                    st.warning(f"Dados insuficientes ou inconsistentes para previsão na janela {janela['nome']}.")
                    continue

                probas = previsao_avancada(X, y, X_pred)
                
                # Mapear os resultados de volta para 'P', 'B', 'T'
                # Assumindo que o order é P, B, T ou B, P, T etc. conforme fit do modelo
                # Uma forma mais robusta seria treinar com LabelEncoder e usar inv_transform
                # Para simplificar aqui, vamos assumir a ordem mais comum ou verificar o class_
                
                # Se y_train contém 'P', 'B', 'T', o modelo provavelmente ordenará as classes.
                # Para robustez, idealmente usar LabelEncoder
                # Exemplo: le = LabelEncoder(); le.fit(['P', 'B', 'T']); y_encoded = le.transform(y)
                # E depois le.inverse_transform([max_idx])
                
                # Por simplicidade, assumindo que as classes são ordenadas como ['B', 'P', 'T'] ou similar
                # ou que o modelo sempre outputa na ordem P, B, T se essas são as classes vistas
                # O mais seguro é verificar as classes do modelo: MODELOS[nome].classes_
                
                # Aqui, estamos assumindo a ordem padrão que XGBoost/RF/MLP geralmente dão
                # (alfabética se não for explicitamente definida por LabelEncoder)
                # Se as classes do modelo forem ['B', 'P', 'T'], probas[0] é para 'B', probas[1] para 'P', probas[2] para 'T'
                # Precisamos saber a ordem para mapear corretamente.
                # Por exemplo, se classes_ == ['B', 'P', 'T']
                
                # Melhoria: usar um mapeamento explícito
                class_map = {
                    'P': 0, 'B': 1, 'T': 2
                } # Esta ordem é arbitrária, precisa ser consistente com o treinamento
                
                # Para ser mais robusto, um LabelEncoder seria ideal no treinamento
                # Ex:
                # from sklearn.preprocessing import LabelEncoder
                # le = LabelEncoder()
                # y_encoded = le.fit_transform(y_train)
                # ...
                # max_idx = np.argmax(probas)
                # lado_pred = le.inverse_transform([max_idx])[0]
                
                # No seu código, está fazendo manualmente:
                # lado_pred = ["P", "B", "T"][max_idx]
                # Isso pressupõe que a classe 0 é 'P', 1 é 'B', 2 é 'T'.
                # Vamos manter assim por enquanto, mas é um ponto de atenção.
                
                max_idx = np.argmax(probas)
                classes_possiveis = ['B', 'P', 'T'] # Assumindo ordem alfabética das classes
                # Ou verificar a ordem exata das classes do modelo se souber
                # Ex: modelo.classes_ para modelos sklearn
                
                # Para maior robustez, especialmente com classificadores multi-classe
                # é essencial garantir que a ordem dos rótulos de saída do predict_proba
                # corresponde à ordem que esperamos (P, B, T).
                # Se os dados de treino y contêm 'P', 'B', 'T', os modelos geralmente
                # atribuirão classes em ordem alfabética (B, P, T).
                # Então, o max_idx 0 seria 'B', 1 seria 'P', 2 seria 'T'.
                
                # Para corrigir:
                # 1. Obtenha as classes do primeiro modelo treinado (se houver)
                #    Ou force a ordem com LabelEncoder durante o treinamento em previsao_avancada
                
                # Vamos simular a ordem alfabética que a maioria dos modelos seguiria
                # Se os seus dados de 'Resultado' têm 'P', 'B', 'T', a ordem das classes
                # pode ser ['B', 'P', 'T'] alfabeticamente.
                # Vamos ajustar a string de mapeamento de acordo.
                
                # Opção 1: Mapeamento manual (se souber a ordem)
                # Por exemplo, se as classes do modelo são sempre ['B', 'P', 'T']
                # classes_labels = ['B', 'P', 'T']
                
                # Opção 2: Usar LabelEncoder para garantir a consistência
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # Ajustar o LabelEncoder com todas as classes possíveis
                le.fit(['P', 'B', 'T'])
                
                # O X_train, y_train de "previsao_avancada" já foi passado.
                # Precisamos que `previsao_avancada` retorne a ordem das classes ou use LabelEncoder internamente.
                # Como `previsao_avancada` já foi chamado, vamos presumir que a ordem das probas é consistente
                # com a ordem das classes aprendidas pelo modelo.
                # Se `y_train` é ['P', 'B', 'T'], `predict_proba` pode retornar a ordem alfabética das classes:
                # [proba_B, proba_P, proba_T].

                # Para ser *realmente* robusto, a função `previsao_avancada` deveria retornar
                # a ordem das classes junto com as probabilidades.
                # Ex: return probas, modelo.classes_
                
                # Por enquanto, mantendo a suposição de que ['P', 'B', 'T'] é a ordem
                # Ou se o modelo interno ordena, verificar isso.
                # Para este cenário, vamos assumir que o `predict_proba` retorna na ordem 'P', 'B', 'T'
                # se P for o primeiro visto, B o segundo, T o terceiro.
                # O mais comum é a ordem alfabética das classes, ou a ordem de ocorrência no y_train.
                # Se y_train contém ['P', 'B', 'T'] na ordem de inserção, predict_proba pode seguir.
                # Mas é um risco.

                # CORREÇÃO: Garanta que a ordem das classes para a previsão seja explícita ou determinada dinamicamente.
                # Vamos simular um caso comum onde as classes são mapeadas alfabeticamente ('B', 'P', 'T')
                # A função `predict_proba` em sklearn geralmente ordena as classes alfabeticamente.
                
                # Para ser seguro, vamos remapear aqui:
                # Obter as classes únicas e ordená-las
                unique_classes = sorted(df_analise["Resultado"].unique())
                
                # Se unique_classes for ['B', 'P', 'T'], então probas[0] é B, probas[1] é P, probas[2] é T
                # Se unique_classes for ['P', 'B', 'T'], então probas[0] é P, probas[1] é B, probas[2] é T
                
                # Vamos assumir que os modelos internos em `previsao_avancada` treinaram em
                # `y_train` e as `predict_proba` retornam as probabilidades na ordem de
                # `modelo.classes_`.
                # Como não passamos `modelo.classes_` para cá, a suposição de `["P", "B", "T"][max_idx]`
                # é frágil. A solução ideal seria modificar `previsao_avancada` para retornar as classes.
                # Por agora, para o propósito da correção, vou manter a sua lógica, mas com um aviso.
                # lado_pred = ["P", "B", "T"][max_idx] # Pode estar errado dependendo da ordem interna do modelo

                # --- Correção de mapeamento de classes ---
                # A forma mais robusta é usar o LabelEncoder.
                # No `previsao_avancada`, os modelos são treinados com `y_train`.
                # O `y_train` contém os rótulos 'P', 'B', 'T'.
                # A ordem dos rótulos em `modelo.classes_` define a ordem das probabilidades em `predict_proba`.
                # Por exemplo, se `modelo.classes_` é `array(['B', 'P', 'T'])`, então `probas[0]` é para 'B', `probas[1]` para 'P', `probas[2]` para 'T'.
                
                # Para simplificar aqui, e sem reestruturar `previsao_avancada` para retornar as classes,
                # vamos fazer uma suposição comum de que o modelo ordena as classes alfabeticamente por padrão.
                # Ou seja, `classes_ordenadas = ['B', 'P', 'T']`
                
                # Vamos criar um LabelEncoder aqui para simular a ordem, mas o ideal seria usar
                # o `LabelEncoder` que foi usado para treinar os modelos em `previsao_avancada`.
                
                # Se `y_train` tem classes ['P', 'B', 'T'], e o modelo as ordena alfabeticamente,
                # as classes seriam ['B', 'P', 'T'].
                # Ou seja, probas[0] é para 'B', probas[1] para 'P', probas[2] para 'T'.
                
                # Para esta correção, vamos assumir que a maioria dos modelos sklearn
                # retorna as probabilidades na ordem alfabética das classes: ['B', 'P', 'T'].
                # Se for este o caso, o mapeamento `["B", "P", "T"][max_idx]` seria o correto.
                
                # No seu código original: `lado_pred = ["P", "B", "T"][max_idx]`
                # Isso implica que a classe 0 é 'P', 1 é 'B', 2 é 'T'.
                # Isso só é verdade se o LabelEncoder for fitado em ['P', 'B', 'T'] e essa for a ordem de classes.
                # Se o modelo naturalmente ordenar, é mais provável que seja ['B', 'P', 'T'].
                
                # Considerando a maior probabilidade de ordem alfabética:
                classes_de_predicao = sorted(df_analise["Resultado"].unique()) # Ex: ['B', 'P', 'T']
                
                if not classes_de_predicao: # Caso não haja resultados no df_analise
                    continue

                # Garantir que temos 3 classes para predict_proba de 3 elementos
                if len(classes_de_predicao) < 3:
                    # Isso significa que o modelo pode não ter visto todas as classes no treinamento da janela
                    # Isso pode levar a erros de predict_proba ou a predições incorretas.
                    # Nesses casos, é melhor pular a previsão.
                    st.warning(f"A janela de análise {janela['nome']} não contém todas as classes (P, B, T). Pulando previsão.")
                    continue
                
                lado_pred = classes_de_predicao[max_idx] # Mapeamento robusto
                
                confianca = probas[max_idx]
                
                if confianca > 0.62:  # Limiar mais alto para confiança
                    todos_padroes.append({
                        "tipo": "PREVISÃO",
                        "lado": lado_pred,
                        "desc": f"Modelo preditivo ({janela['nome']}) sugere {lado_pred} (conf: {confianca*100:.1f}%)",
                        "peso": min(4.0, confianca * 6) * peso_janela,
                        "janela": janela["nome"]
                    })
            except Exception as e:
                st.warning(f"Erro na previsão com modelo para janela {janela['nome']}: {e}")
    
    # 4. Análise de Probabilidade Condicional (histórico completo)
    if len(df_completo) > 100: # Garantir dados suficientes
        try:
            # Player ganha quando soma > 8
            prob_p_gt_8 = calcular_probabilidade_condicional(
                df_completo,
                "Resultado == 'P'",
                "Player > 8"
            )
            if prob_p_gt_8 > 58:  # Limiar mais alto
                todos_padroes.append({
                    "tipo": "PROBABILIDADE",
                    "lado": "P",
                    "desc": f"Prob histórica: Player ganha {prob_p_gt_8:.1f}% quando soma Player > 8",
                    "peso": min(3.0, (prob_p_gt_8-50)/8),
                    "janela": "Histórico"
                })
                
            # Banker ganha quando soma > 9
            prob_b_gt_9 = calcular_probabilidade_condicional(
                df_completo,
                "Resultado == 'B'",
                "Banker > 9"
            )
            if prob_b_gt_9 > 58:
                todos_padroes.append({
                    "tipo": "PROBABILIDADE",
                    "lado": "B",
                    "desc": f"Prob histórica: Banker ganha {prob_b_gt_9:.1f}% quando soma Banker > 9",
                    "peso": min(3.0, (prob_b_gt_9-50)/8),
                    "janela": "Histórico"
                })
                
            # Tie quando diferença pequena
            prob_t_diff_le_1 = calcular_probabilidade_condicional(
                df_completo,
                "Resultado == 'T'",
                "abs(Player - Banker) <= 1"
            )
            if prob_t_diff_le_1 > 15:  # Probabilidade natural ~10%
                todos_padroes.append({
                    "tipo": "PROBABILIDADE",
                    "lado": "T",
                    "desc": f"Prob histórica: Tie ocorre em {prob_t_diff_le_1:.1f}% quando diferença <=1",
                    "peso": min(3.0, prob_t_diff_le_1/6),
                    "janela": "Histórico"
                })
        except Exception as e:
            st.warning(f"Erro na análise de probabilidade condicional: {e}")
            pass
    
    # 5. Padrões de Sequência
    if not df_completo.empty: # Checar se df_completo não está vazio
        resultados = df_completo["Resultado"].values
        if len(resultados) > 10:
            # Detecção de sequências P-B-P-B
            padrao_alternancia = 0
            # Corrigir o loop para evitar IndexError ao acessar i-3, i-2, i-1
            for i in range(3, len(resultados)): # Começa de 3 para poder acessar i-3
                if (resultados[i-3] == 'P' and resultados[i-2] == 'B' and
                    resultados[i-1] == 'P' and resultados[i] == 'B'):
                    padrao_alternancia += 1
            
            if padrao_alternancia >= 2:
                todos_padroes.append({
                    "tipo": "SEQUÊNCIA",
                    "lado": "AMBOS", # Ou 'P' ou 'B' dependendo do final da sequência, ou AMBOS se for um padrão geral
                    "desc": f"Padrão de alternância P-B-P-B detectado {padrao_alternancia} vezes",
                    "peso": 2.5,
                    "janela": "Longo"
                })
    
    return todos_padroes

def gerar_recomendacao(padroes):
    if not padroes:
        return "AGUARDAR", 15, "Sem padrões detectados. Aguarde mais dados.", "warning"
    
    # Agrupar padrões por lado
    scores = {"P": 0.0, "B": 0.0, "T": 0.0}
    detalhes = {"P": [], "B": [], "T": []}
    
    for padrao in padroes:
        lado = padrao["lado"]
        peso = padrao["peso"]
        
        if lado in scores:
            scores[lado] += peso
            detalhes[lado].append(f"{padrao['tipo']}: {padrao['desc']}")
        elif lado == "AMBOS": # Se o padrão é "AMBOS", divide o peso ou aplica a ambos
            scores["P"] += peso / 2 # Divide o peso entre P e B
            scores["B"] += peso / 2
            detalhes["P"].append(f"{padrao['tipo']}: {padrao['desc']} (afeta P e B)")
            detalhes["B"].append(f"{padrao['tipo']}: {padrao['desc']} (afeta P e B)")
    
    # Calcular confiança
    total_score = sum(scores.values())
    if total_score == 0:
        return "AGUARDAR", 10, "Padrões sem força significativa", "warning"
    
    # Normaliza as confianças para somar 100%
    confiancas = {lado: min(100, int(score / total_score * 100)) for lado, score in scores.items()}
    
    # Determinar recomendação com limiares mais altos
    # Usar max() com o lambda para pegar a chave (lado) com o maior valor (score)
    max_lado = max(scores, key=scores.get)
    max_score = scores[max_lado]
    
    # Limiares de decisão mais rigorosos
    if max_score > 6.0:
        acao = f"APOSTAR FORTE NO {'PLAYER' if max_lado == 'P' else 'BANKER' if max_lado == 'B' else 'TIE'}"
        tipo = "success"
        conf = confiancas[max_lado]
        detalhe = f"**Convergência poderosa de padrões** ({max_score:.1f} pontos):\n- " + "\n- ".join(detalhes[max_lado])
    elif max_score > 4.0:
        acao = f"APOSTAR NO {'PLAYER' if max_lado == 'P' else 'BANKER' if max_lado == 'B' else 'TIE'}"
        tipo = "success"
        conf = confiancas[max_lado]
        detalhe = f"**Forte convergência de padrões** ({max_score:.1f} pontos):\n- " + "\n- ".join(detalhes[max_lado])
    elif max_score > 2.5:
        acao = f"CONSIDERAR {'PLAYER' if max_lado == 'P' else 'BANKER' if max_lado == 'B' else 'TIE'}"
        tipo = "warning"
        conf = confiancas[max_lado]
        detalhe = f"**Sinal moderado** ({max_score:.1f} pontos):\n- " + "\n- ".join(detalhes[max_lado])
    else:
        acao = "AGUARDAR"
        tipo = "warning"
        # Quando AGUARDAR, a confiança é o quão perto os scores estão, ou simplesmente baixa
        # Aqui, vamos definir a confiança como a maior confiança, mas a ação é aguardar.
        # Poderíamos também usar a entropia ou algo similar para indicar incerteza.
        # Para simplificar, mantemos 100 - max(confianças) como um proxy de "não há um lado claro".
        conf = 100 - confiancas[max_lado] # Representa a incerteza ou a falta de um sinal dominante
        detalhe = "**Sinais fracos ou conflitantes**. Aguarde confirmação:\n- " + "\n- ".join(
            [f"{lado}: {score:.1f} pts" for lado, score in scores.items()])
    
    return acao, conf, detalhe, tipo

def estrategia_simples(df):
    """Estratégia básica baseada no último resultado"""
    if df.empty:
        return None
    ultimo = df.iloc[-1]
    if ultimo['Player'] > ultimo['Banker']:
        return 'P'
    elif ultimo['Banker'] > ultimo['Player']:
        return 'B'
    else:
        return 'T'

def estrategia_ia(df):
    """Estratégia avançada usando detecção de padrões"""
    if df.empty or len(df) < 20: # Garantir dados mínimos para detecção de padrões
        return None
    padroes = detectar_padroes_avancados(df)
    acao, _, _, _ = gerar_recomendacao(padroes)
    
    if "PLAYER" in acao:
        return 'P'
    elif "BANKER" in acao:
        return 'B'
    elif "TIE" in acao:
        return 'T'
    else:
        return None  # Não aposta quando é AGUARDAR

def executar_backtesting(df_completo, estrategia_func, tamanho_janela=20):
    if len(df_completo) < tamanho_janela:
        st.warning(f"Dados insuficientes para backtesting com janela de {tamanho_janela}. Necessário {tamanho_janela} registros.")
        return {} # Retorna dicionário vazio se não há dados suficientes

    resultados = []
    saldo = 1000.0 # Usar float para saldo
    apostas = [] # 1 para vitória, 0 para derrota
    detalhes = []
    acoes = [] # Recomendação de aposta
    
    # Criar DataFrame com as colunas analíticas necessárias para a estratégia_ia
    df_analise_completa = df_completo.copy()
    df_analise_completa['Diferenca'] = abs(df_analise_completa['Player'] - df_analise_completa['Banker'])
    df_analise_completa['SomaTotal'] = df_analise_completa['Player'] + df_analise_completa['Banker']
    df_analise_completa['Vencedor'] = np.where(
        df_analise_completa['Resultado'] == 'P', 'Player',
        np.where(df_analise_completa['Resultado'] == 'B', 'Banker', 'Tie')
    )

    for i in range(tamanho_janela, len(df_analise_completa)):
        # Garantir que a fatia de dados é um DataFrame para a estratégia_ia
        dados_janela = df_analise_completa.iloc[i-tamanho_janela:i].copy()
        
        recomendacao = estrategia_func(dados_janela)
        
        # Valor da aposta base
        aposta_valor_player_banker = 50.0
        aposta_valor_tie = 80.0

        if recomendacao is None:  # Quando estratégia recomenda AGUARDAR
            detalhes.append({
                "jogo": i,
                "aposta": "Nenhuma",
                "resultado": df_analise_completa.iloc[i]['Resultado'],
                "ganho": 0.0,
                "saldo": saldo
            })
            continue # Pula para a próxima iteração
            
        resultado_real = df_analise_completa.iloc[i]['Resultado']
        
        if recomendacao == resultado_real:
            if recomendacao == 'T':
                ganho = aposta_valor_tie * 8.0 # Pagamento do Tie é geralmente 8:1
                saldo += ganho
            else:
                ganho = aposta_valor_player_banker * 1.0 # Pagamento P/B é 1:1
                saldo += ganho
            apostas.append(1)  # Vitória
        else:
            if recomendacao == 'T':
                perda = aposta_valor_tie
                saldo -= perda
            else:
                perda = aposta_valor_player_banker
                saldo -= perda
            apostas.append(0)  # Derrota
            
        detalhes.append({
            "jogo": i,
            "aposta": recomendacao,
            "resultado": resultado_real,
            "ganho": ganho if recomendacao == resultado_real else -perda,
            "saldo": saldo
        })
        acoes.append(recomendacao)
    
    # Calcular métricas
    win_rate = np.mean(apostas) * 100 if apostas else 0
    retorno = ((saldo - 1000) / 1000) * 100 if 1000 != 0 else 0 # Evitar divisão por zero
    
    return {
        "saldo_final": saldo,
        "win_rate": win_rate,
        "retorno_percent": retorno,
        "detalhes": detalhes,
        "acoes": acoes
    }

# --- Interface Premium ---
st.markdown("""
<div style="text-align:center; margin-bottom:30px;">
    <h1 style="color:#ffc107; font-size:2.5em;">BAC BO PREDICTOR PRO</h1>
    <p style="font-size:1.2em;">Sistema de análise preditiva com algoritmos de Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# --- Entrada de Dados Premium ---
with st.expander("🎮 ENTRADA DE DADOS", expanded=True):
    col1, col2, col3, col4 = st.columns([1,1,1,0.8])
    with col1:
        player_soma = st.number_input("Soma Player (2-12)", min_value=2, max_value=12, value=7, key="player_soma_input")
    with col2:
        banker_soma = st.number_input("Soma Banker (2-12)", min_value=2, max_value=12, value=7, key="banker_soma_input")
    with col3:
        resultado_op = st.selectbox("Resultado", ['P', 'B', 'T'], key="resultado_select")
    with col4:
        st.write("") # Espaço para alinhar o botão
        st.write("")
        if st.button("➕ ADICIONAR", use_container_width=True, type="primary"):
            st.session_state.historico_dados.append((player_soma, banker_soma, resultado_op))
            st.rerun() # Use st.rerun() para atualizar o Streamlit

# --- Histórico com Visualização Premium ---
st.subheader("📜 HISTÓRICO DE RESULTADOS")
if st.session_state.historico_dados:
    # Criar DataFrame a partir do histórico para análise e exibição
    df_historico = pd.DataFrame(
        st.session_state.historico_dados,
        columns=["Player", "Banker", "Resultado"]
    )
    
    # Adicionar colunas analíticas (fazer isso uma vez para o DataFrame principal)
    df_historico['Diferenca'] = abs(df_historico['Player'] - df_historico['Banker'])
    df_historico['SomaTotal'] = df_historico['Player'] + df_historico['Banker']
    df_historico['Vencedor'] = np.where(
        df_historico['Resultado'] == 'P', 'Player',
        np.where(df_historico['Resultado'] == 'B', 'Banker', 'Tie')
    )
    
    # Exibir tabela com estilo
    st.dataframe(df_historico.tail(20).style
        .background_gradient(subset=['Player', 'Banker'], cmap='YlGnBu')
        .applymap(lambda x: 'color: #1f77b4; font-weight: bold' if x == 'P' else # Azul para Player
                 ('color: #ff7f0e; font-weight: bold' if x == 'B' else # Laranja para Banker
                  'color: #2ca02c; font-weight: bold'), # Verde para Tie
                subset=['Resultado']),
        use_container_width=True, height=450)
    
    # Controles do histórico
    col_hist1, col_hist2, col_hist3 = st.columns([1,1,2])
    with col_hist1:
        if st.button("🗑️ REMOVER ÚLTIMO", use_container_width=True):
            if st.session_state.historico_dados:
                st.session_state.historico_dados.pop()
                st.rerun()
    with col_hist2:
        if st.button("🧹 LIMPAR TUDO", use_container_width=True, type="secondary"):
            st.session_state.historico_dados = []
            st.session_state.padroes_detectados = []
            st.session_state.historico_recomendacoes = [] # Limpar também as recomendações
            st.session_state.backtest_results = {} # Limpar resultados de backtest
            st.rerun()
    with col_hist3:
        last = df_historico.iloc[-1] if not df_historico.empty else {}
        st.info(f"🔢 Total: {len(df_historico)} | Último: {last.get('Player', '')}-{last.get('Banker', '')}-{last.get('Resultado', '')}")
else:
    st.warning("⚠️ Nenhum dado no histórico. Adicione resultados para iniciar a análise.")

# --- Entrada em Massa Premium ---
with st.expander("📥 IMPORTAR DADOS EM MASSA", expanded=False):
    historico_input_mass = st.text_area("Cole múltiplas linhas (1 linha = Player,Banker,Resultado)", height=150,
                                         placeholder="Ex: 7,5,P\n8,8,T\n6,9,B")
    
    if st.button("🚀 PROCESSAR DADOS", use_container_width=True, type="primary"):
        linhas = [linha.strip() for linha in historico_input_mass.split("\n") if linha.strip()]
        novos_dados = []
        erros = []
        
        for i, linha in enumerate(linhas, 1):
            try:
                partes = [p.strip() for p in linha.split(',')]
                if len(partes) != 3: # Deve ter exatamente 3 partes
                    erros.append(f"Linha {i}: Formato inválido (esperado: Player,Banker,Resultado). Encontrado {len(partes)} partes.")
                    continue
                
                p = int(partes[0])
                b = int(partes[1])
                r = partes[2].upper()
                
                # Validações mais robustas
                if not (2 <= p <= 12):
                    erros.append(f"Linha {i}: Soma Player inválida ({p}) - deve ser 2-12")
                if not (2 <= b <= 12):
                    erros.append(f"Linha {i}: Soma Banker inválida ({b}) - deve ser 2-12")
                if r not in ['P', 'B', 'T']:
                    erros.append(f"Linha {i}: Resultado inválido ({r}) - deve ser P, B ou T")
                
                # Adiciona apenas se não houver erros específicos para esta linha
                if not any(erro.startswith(f"Linha {i}") for erro in erros):
                    novos_dados.append((p, b, r))
            except ValueError:
                erros.append(f"Linha {i}: Valores numéricos inválidos ou ausentes.")
            except Exception as e:
                erros.append(f"Linha {i}: Erro de processamento - {str(e)}")
        
        if novos_dados: # Adiciona dados válidos se houver algum
            st.session_state.historico_dados.extend(novos_dados)
            st.success(f"✅ {len(novos_dados)} novos registros adicionados com sucesso!")
            st.rerun()
        if erros: # Exibe todos os erros de uma vez
            st.error("❌ Erros encontrados ao processar os dados:")
            for erro in erros:
                st.error(f"- {erro}")
        elif not novos_dados and not erros:
            st.info("Nenhuma nova linha válida para adicionar.")
    
    # Exportar dados
    if st.session_state.historico_dados:
        csv = pd.DataFrame(st.session_state.historico_dados, 
                          columns=["Player", "Banker", "Resultado"]).to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="💾 EXPORTAR DADOS (CSV)",
            data=csv,
            file_name=f"bacbo_historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", # Adiciona hora para unicidade
            mime='text/csv',
            use_container_width=True
        )

# --- ANÁLISE E RECOMENDAÇÃO ---
if st.session_state.historico_dados:
    # Garante que df_analise é sempre criado a partir de historico_dados para consistência
    df_analise = pd.DataFrame(
        st.session_state.historico_dados,
        columns=["Player", "Banker", "Resultado"]
    )
    
    # Adicionar colunas analíticas ao df_analise (essencial para as funções de análise)
    df_analise['Diferenca'] = abs(df_analise['Player'] - df_analise['Banker'])
    df_analise['SomaTotal'] = df_analise['Player'] + df_analise['Banker']
    df_analise['Vencedor'] = np.where(
        df_analise['Resultado'] == 'P', 'Player',
        np.where(df_analise['Resultado'] == 'B', 'Banker', 'Tie')
    )

    if len(df_analise) > 10: # Mínimo de dados para começar a analisar
        with st.spinner("🔍 Analisando padrões e executando modelos de IA..."):
            # Detecta padrões
            st.session_state.padroes_detectados = detectar_padroes_avancados(df_analise)
            
            # Gera recomendação
            acao, conf, detalhe, tipo = gerar_recomendacao(st.session_state.padroes_detectados)
            
            # Armazena histórico de recomendações
            # Verificar se a recomendação atual é diferente da última para evitar duplicatas repetitivas
            if not st.session_state.historico_recomendacoes or \
               st.session_state.historico_recomendacoes[-1]['acao'] != acao or \
               st.session_state.historico_recomendacoes[-1]['confianca'] != conf: # Ajustar critério de duplicata
                
                st.session_state.historico_recomendacoes.append({
                    "timestamp": datetime.now(),
                    "acao": acao,
                    "confianca": conf,
                    "tipo": tipo,
                    "detalhes": detalhe
                })
            
            # Exibe recomendação
            st.markdown(f"<div class='stAlert alert-{tipo}'>{acao} (Confiança: {conf}%)</div>", unsafe_allow_html=True)
            st.markdown(f"**Detalhes da Análise:**\n{detalhe}")
            
            # Atualiza o estado dos modelos (apenas se for relevante para a exibição)
            st.session_state.modelos_treinados = True # Isso é mais uma flag conceitual
            st.session_state.ultimo_treinamento = datetime.now() # Reflete a última vez que a análise rodou

            # Exibir padrões detectados
            st.subheader("📊 Padrões Detectados")
            if st.session_state.padroes_detectados:
                df_padroes = pd.DataFrame(st.session_state.padroes_detectados)
                st.dataframe(df_padroes[['tipo', 'lado', 'desc', 'peso', 'janela']], use_container_width=True)
            else:
                st.warning("Nenhum padrão significativo detectado com a base de dados atual.")
                
            # --- Visualizações Gráficas ---
            tab1, tab2, tab3 = st.tabs(["📈 Tendências", "📊 Distribuição", "🧠 Modelos"])
            
            with tab1:
                # Gráfico de tendência das somas
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_analise.index,
                    y=df_analise['Player'],
                    name='Player',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=df_analise.index,
                    y=df_analise['Banker'],
                    name='Banker',
                    line=dict(color='#ff7f0e', width=3)
                ))
                fig.update_layout(
                    title='Evolução das Somas Player vs Banker',
                    xaxis_title='Rodada',
                    yaxis_title='Soma',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de diferenças
                fig = px.bar(
                    df_analise,
                    x=df_analise.index,
                    y='Diferenca',
                    title='Diferença entre Player e Banker',
                    color='Resultado',
                    color_discrete_map={'P': '#1f77b4', 'B': '#ff7f0e', 'T': '#2ca02c'}
                )
                fig.update_layout(
                    template='plotly_dark',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                # Histograma comparativo
                fig = px.histogram(
                    df_analise,
                    x=['Player', 'Banker'],
                    nbins=11,
                    barmode='overlay',
                    opacity=0.7,
                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                )
                fig.update_layout(
                    title='Distribuição das Somas',
                    xaxis_title='Soma',
                    yaxis_title='Frequência',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de pizza de resultados
                result_counts = df_analise['Resultado'].value_counts()
                fig = px.pie(
                    result_counts,
                    values=result_counts.values,
                    names=result_counts.index,
                    title='Distribuição de Resultados',
                    color=result_counts.index,
                    color_discrete_map={'P': '#1f77b4', 'B': '#ff7f0e', 'T': '#2ca02c'}
                )
                fig.update_layout(
                    template='plotly_dark',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                # Treinar e avaliar modelos
                # Adicionado `st.session_state.modelos_treinados` para controle
                if len(df_analise) > 50: # Mínimo de dados para train_test_split
                    st.subheader("Desempenho dos Modelos")
                    X = df_analise[["Player", "Banker"]]
                    y = df_analise["Resultado"]
                    
                    # Dividir dados
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y # Adicionado stratify para balancear classes
                        )
                    except ValueError as e:
                        st.warning(f"Não foi possível dividir os dados para treinamento do modelo: {e}. Pode ser que uma classe tenha poucas amostras.")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42 # Remove stratify se der erro
                        )

                    # Aplicar StandardScaler antes de treinar modelos
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    resultados_modelos = []
                    for nome, modelo in MODELOS.items():
                        try:
                            start_time = time.time()
                            # Clone o modelo antes de treinar para garantir um novo estado em cada iteração
                            m = modelo.__class__(**modelo.get_params())
                            m.fit(X_train_scaled, y_train)
                            y_pred = m.predict(X_test_scaled)
                            acc = accuracy_score(y_test, y_pred)
                            resultados_modelos.append({
                                "Modelo": nome,
                                "Acurácia": f"{acc*100:.1f}%",
                                "Tempo (s)": f"{time.time()-start_time:.3f}",
                                "Status": "✅" if acc > 0.6 else ("⚠️" if acc > 0.5 else "❌") # Critério mais claro
                            })
                        except Exception as e:
                            resultados_modelos.append({
                                "Modelo": nome,
                                "Acurácia": "ERRO",
                                "Tempo (s)": "N/A",
                                "Status": "❌"
                            })
                    
                    df_resultados = pd.DataFrame(resultados_modelos)
                    st.dataframe(df_resultados, use_container_width=True)
                    
                    # Relatório de classificação
                    if st.checkbox("Mostrar relatório detalhado do melhor modelo", key="show_report_checkbox"):
                        if not df_resultados.empty and 'Acurácia' in df_resultados.columns:
                            # Converte 'Acurácia' para numérico para encontrar o máximo
                            df_resultados['Acurácia_Num'] = df_resultados['Acurácia'].str.rstrip('%').astype(float) / 100
                            melhor_modelo_nome = df_resultados.loc[df_resultados['Acurácia_Num'].idxmax()]['Modelo']
                            
                            modelo_obj = MODELOS[melhor_modelo_nome]
                            # Retreinar o modelo para ter a instância exata se necessário, ou usar a já treinada se o cache estiver ativo
                            m = modelo_obj.__class__(**modelo_obj.get_params()) # Clone para garantir
                            m.fit(X_train_scaled, y_train)
                            y_pred = m.predict(X_test_scaled)
                            
                            st.subheader(f"Relatório de Classificação - {melhor_modelo_nome}")
                            try:
                                report = classification_report(y_test, y_pred, output_dict=True)
                                df_report = pd.DataFrame(report).transpose()
                                st.dataframe(df_report.style.highlight_max(axis=0, color='#2a8c55'), use_container_width=True)
                            except ValueError as e:
                                st.warning(f"Não foi possível gerar o relatório de classificação detalhado: {e}. Pode ser que algumas classes não estejam presentes no conjunto de teste.")
                        else:
                            st.info("Nenhum resultado de modelo disponível para gerar relatório.")
                    
                    # Informação sobre atualização
                    if st.session_state.ultimo_treinamento:
                        st.caption(f"Último treinamento dos modelos: {st.session_state.ultimo_treinamento.strftime('%d/%m/%Y %H:%M:%S')}")
                else:
                    st.warning("Dados insuficientes para treinar modelos de Machine Learning (mínimo 50 registros).")
            
            # --- Backtesting ---
            st.subheader("🧪 Teste de Estratégia")
            col_strat, col_size, col_run = st.columns([2, 1, 1])
            with col_strat:
                estrategia_selecionada = st.selectbox(
                    "Selecione a Estratégia",
                    ["Simples (Último Resultado)", "IA (Recomendação Inteligente)"],
                    index=0,
                    key="strategy_select"
                )
            with col_size:
                tamanho_janela_backtest = st.selectbox("Janela de Análise para Backtesting", [20, 30, 50, 100], index=0,
                                                       key="backtest_window_size")
            with col_run:
                st.write("") # Espaço para alinhar o botão
                if st.button("🔁 Executar Backtesting", use_container_width=True):
                    if len(df_analise) >= tamanho_janela_backtest + 1: # Precisa de dados para a janela + 1 para o resultado
                        with st.spinner("Executando simulação histórica..."):
                            # Seleciona estratégia
                            if "Simples" in estrategia_selecionada:
                                resultados_backtest = executar_backtesting(df_analise, estrategia_simples, tamanho_janela_backtest)
                            else: # Estratégia IA
                                resultados_backtest = executar_backtesting(df_analise, estrategia_ia, tamanho_janela_backtest)
                            
                            st.session_state.backtest_results = resultados_backtest
                            
                            if resultados_backtest: # Verifica se há resultados
                                # Exibir resultados
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Saldo Inicial", f"R$1.000,00") # Sempre 1000
                                col2.metric("Saldo Final", f"R${resultados_backtest['saldo_final']:,.2f}",
                                            delta=f"{resultados_backtest['saldo_final']-1000:,.2f}")
                                col3.metric("Win Rate", f"{resultados_backtest['win_rate']:.1f}%")
                                st.metric("Retorno Total", f"{resultados_backtest['retorno_percent']:.1f}%",
                                          delta=f"{resultados_backtest['retorno_percent']:.1f}%")
                                
                                # Gráfico de evolução do saldo
                                df_evolucao = pd.DataFrame(resultados_backtest['detalhes'])
                                if not df_evolucao.empty:
                                    fig = px.line(
                                        df_evolucao,
                                        x='jogo',
                                        y='saldo',
                                        title='Evolução do Saldo no Backtesting',
                                        markers=True
                                    )
                                    fig.update_layout(
                                        xaxis_title='Rodada',
                                        yaxis_title='Saldo (R$)',
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Nenhuma aposta foi feita durante o backtesting ou os dados são insuficientes.")

                                # Análise de acertos
                                st.subheader("Análise de Desempenho do Backtesting")
                                if resultados_backtest['acoes']:
                                    acoes_counts = pd.Series(resultados_backtest['acoes']).value_counts()
                                    fig = px.bar(
                                        acoes_counts,
                                        x=acoes_counts.index,
                                        y=acoes_counts.values,
                                        title='Distribuição de Apostas no Backtesting',
                                        labels={'x': 'Aposta Recomendada', 'y': 'Quantidade'},
                                        color=acoes_counts.index,
                                        color_discrete_map={'P': '#1f77b4', 'B': '#ff7f0e', 'T': '#2ca02c'}
                                    )
                                    fig.update_layout(template='plotly_dark', showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Nenhuma aposta recomendada durante o backtesting para analisar.")
                            else:
                                st.error("Erro ao executar backtesting ou dados insuficientes após filtragem.")
                    else:
                        st.warning(f"Necessário mínimo de {tamanho_janela_backtest + 1} registros para backtesting.")
            
            # Exibir resultados de backtesting se existirem
            if st.session_state.backtest_results:
                if 'saldo_final' in st.session_state.backtest_results:
                    st.info(f"Último backtesting ({estrategia_selecionada}): "
                            f"Saldo Final R${st.session_state.backtest_results['saldo_final']:,.2f} | "
                            f"Win Rate {st.session_state.backtest_results['win_rate']:.1f}% | "
                            f"Retorno {st.session_state.backtest_results['retorno_percent']:.1f}%")
                else:
                    st.info("Nenhum resultado de backtesting disponível. Execute o backtesting.")
            
            # --- Histórico de Recomendações ---
            st.subheader("🕒 Histórico de Recomendações")
            if st.session_state.historico_recomendacoes:
                df_recomendacoes = pd.DataFrame(st.session_state.historico_recomendacoes)
                df_recomendacoes['timestamp'] = df_recomendacoes['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') # Formatar para exibição
                df_recomendacoes = df_recomendacoes.sort_values('timestamp', ascending=False)
                # Selecionar colunas para exibição clara
                st.dataframe(df_recomendacoes[['timestamp', 'acao', 'confianca', 'tipo', 'detalhes']].head(10), use_container_width=True)
            else:
                st.info("Nenhuma recomendação registrada ainda.")
                
    else:
        st.info("ℹ️ Adicione mais dados para ativar a análise preditiva (mínimo 10 rodadas para análise inicial).")
else:
    st.info("ℹ️ Adicione dados para ativar a análise preditiva.")

# --- Rodapé ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #aaa; font-size: 0.9em; padding: 20px;">
    BAC BO PREDICTOR PRO v2.1 | Sistema de Análise Preditiva | 
    Desenvolvido com Streamlit e Machine Learning | © 2023-2024
</div>
""", unsafe_allow_html=True)

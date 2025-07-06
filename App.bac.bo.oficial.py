import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Configuração Avançada da Página ---
st.set_page_config(page_title="Bac Bo Inteligente Premium", layout="wide", initial_sidebar_state="expanded")
st.title("🎲 Analisador de Alta Precisão - Bac Bo Evolution (v4.0)")

# Estilos CSS otimizados
st.markdown("""
<style>
    /* Estilos aprimorados para destaque visual */
    .stAlert {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        font-size: 1.3em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .alert-success { background: linear-gradient(135deg, #28a745, #1e7e34); }
    .alert-danger { background: linear-gradient(135deg, #dc3545, #bd2130); }
    .alert-warning { background: linear-gradient(135deg, #ffc107, #e0a800); }
    
    /* Melhorias na tabela */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    /* Destaque para as métricas */
    .stMetric {
        background-color: #2e2f3a;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Inicialização do Session State ---
if 'historico_dados' not in st.session_state:
    st.session_state.historico_dados = []
    st.session_state.padroes_detectados = []
    st.session_state.modelo_treinado = None

# --- Entrada de Dados (Interface Aprimorada) ---
st.markdown("""
### 📥 Entrada de Resultados
Adicione resultados no formato: **Player,Banker,Resultado**  
Exemplo: `11,4,P` para Player, `7,11,B` para Banker, `6,6,T` para Tie
""")

with st.expander("🔍 Adicionar Resultados Individualmente", expanded=True):
    col1, col2, col3, col4 = st.columns([1,1,1,0.5])
    with col1:
        player_soma = st.number_input("Soma Player (2-12)", min_value=2, max_value=12, value=7, key="player_soma_input")
    with col2:
        banker_soma = st.number_input("Soma Banker (2-12)", min_value=2, max_value=12, value=7, key="banker_soma_input")
    with col3:
        resultado_op = st.selectbox("Resultado", ['P', 'B', 'T'], key="resultado_select")
    with col4:
        st.write("")
        st.write("")
        if st.button("➕ Adicionar", use_container_width=True):
            st.session_state.historico_dados.append((player_soma, banker_soma, resultado_op))
            st.rerun()

# --- Histórico com Visualização Avançada ---
st.subheader("📋 Histórico de Resultados")
if st.session_state.historico_dados:
    df_historico = pd.DataFrame(
        st.session_state.historico_dados,
        columns=["Player", "Banker", "Resultado"]
    )
    
    # Adicionar colunas analíticas
    df_historico['Diferenca'] = abs(df_historico['Player'] - df_historico['Banker'])
    df_historico['SomaTotal'] = df_historico['Player'] + df_historico['Banker']
    df_historico['Vencedor'] = np.where(
        df_historico['Resultado'] == 'P', 'Player',
        np.where(df_historico['Resultado'] == 'B', 'Banker', 'Tie')
    )
    
    # Exibir tabela com métricas
    st.dataframe(df_historico.tail(20).style
        .background_gradient(subset=['Player', 'Banker'], cmap='YlGnBu')
        .applymap(lambda x: 'color: blue' if x == 'P' else ('color: red' if x == 'B' else 'color: green'), 
                subset=['Resultado']),
        use_container_width=True, height=400)
    
    # Controles do histórico
    col_hist1, col_hist2, col_hist3 = st.columns([1,1,2])
    with col_hist1:
        if st.button("🗑️ Remover Último", use_container_width=True):
            if st.session_state.historico_dados:
                st.session_state.historico_dados.pop()
                st.rerun()
    with col_hist2:
        if st.button("🧹 Limpar Tudo", use_container_width=True, type="secondary"):
            st.session_state.historico_dados = []
            st.session_state.padroes_detectados = []
            st.rerun()
    with col_hist3:
        st.info(f"Total de registros: {len(df_historico)} | Último: {df_historico.iloc[-1]['Player']}-{df_historico.iloc[-1]['Banker']}-{df_historico.iloc[-1]['Resultado']}")
else:
    st.warning("Nenhum dado no histórico. Adicione resultados para iniciar a análise.")

# --- Entrada em Massa com Validação Avançada ---
with st.expander("📤 Importar Dados em Massa", expanded=False):
    historico_input_mass = st.text_area("Cole múltiplas linhas (1 linha = Player,Banker,Resultado)", height=150)
    
    if st.button("Processar Dados em Massa", use_container_width=True):
        linhas = [linha.strip() for linha in historico_input_mass.split("\n") if linha.strip()]
        novos_dados = []
        erros = []
        
        for i, linha in enumerate(linhas, 1):
            try:
                partes = [p.strip() for p in linha.split(',')]
                if len(partes) < 3:
                    erros.append(f"Linha {i}: Formato inválido (esperado: Player,Banker,Resultado)")
                    continue
                
                p = int(partes[0])
                b = int(partes[1])
                r = partes[2].upper()
                
                if not (2 <= p <= 12):
                    erros.append(f"Linha {i}: Soma Player inválida ({p}) - deve ser 2-12")
                if not (2 <= b <= 12):
                    erros.append(f"Linha {i}: Soma Banker inválida ({b}) - deve ser 2-12")
                if r not in ['P', 'B', 'T']:
                    erros.append(f"Linha {i}: Resultado inválido ({r}) - deve ser P, B ou T")
                
                if not erros or not any(f"Linha {i}" in e for e in erros):
                    novos_dados.append((p, b, r))
            except Exception as e:
                erros.append(f"Linha {i}: Erro de processamento - {str(e)}")
        
        if erros:
            for erro in erros:
                st.error(erro)
        else:
            st.session_state.historico_dados.extend(novos_dados)
            st.success(f"✅ {len(novos_dados)} linhas adicionadas com sucesso!")
            st.rerun()

# --- Verificação de Dados ---
if not st.session_state.historico_dados:
    st.warning("Adicione dados para iniciar a análise!")
    st.stop()

df = pd.DataFrame(
    st.session_state.historico_dados,
    columns=["Player", "Banker", "Resultado"]
)

# --- ALGORITMOS AVANÇADOS DE ANÁLISE ---

# 1. Sistema de Detecção de Padrões com Pesos Dinâmicos
def detectar_padroes_avancados(df_analise):
    padroes = []
    n = len(df_analise)
    
    if n < 3:
        return padroes
    
    # 1. Sequências de Somas Crescentes/Decrescentes
    player_trend = np.polyfit(range(n), df_analise["Player"], 1)[0]
    banker_trend = np.polyfit(range(n), df_analise["Banker"], 1)[0]
    
    if player_trend > 0.15:
        padroes.append({"tipo": "TENDÊNCIA", "lado": "P", "desc": f"Soma Player em alta (slope: {player_trend:.2f})", "peso": 1.5})
    elif player_trend < -0.15:
        padroes.append({"tipo": "TENDÊNCIA", "lado": "P", "desc": f"Soma Player em queda (slope: {player_trend:.2f})", "peso": 1.5})
    
    if banker_trend > 0.15:
        padroes.append({"tipo": "TENDÊNCIA", "lado": "B", "desc": f"Soma Banker em alta (slope: {banker_trend:.2f})", "peso": 1.5})
    elif banker_trend < -0.15:
        padroes.append({"tipo": "TENDÊNCIA", "lado": "B", "desc": f"Soma Banker em queda (slope: {banker_trend:.2f})", "peso": 1.5})
    
    # 2. Padrões de Repetição de Somas
    player_counts = Counter(df_analise["Player"])
    banker_counts = Counter(df_analise["Banker"])
    
    for soma, count in player_counts.items():
        if count >= max(3, n*0.3):  # Pelo menos 3 ou 30% dos resultados
            padroes.append({
                "tipo": "REPETIÇÃO", 
                "lado": "P", 
                "desc": f"Soma Player {soma} repetida {count} vezes",
                "peso": min(2.0, count*0.5)
            })
    
    for soma, count in banker_counts.items():
        if count >= max(3, n*0.3):
            padroes.append({
                "tipo": "REPETIÇÃO", 
                "lado": "B", 
                "desc": f"Soma Banker {soma} repetida {count} vezes",
                "peso": min(2.0, count*0.5)
            })
    
    # 3. Padrões de Resultados (Streaks e Alternâncias)
    resultados = df_analise["Resultado"].tolist()
    
    # Detecção de streaks
    current_streak = 1
    last_result = resultados[0]
    
    for i in range(1, n):
        if resultados[i] == last_result and resultados[i] != 'T':
            current_streak += 1
        else:
            if current_streak >= 3:
                padroes.append({
                    "tipo": "STREAK", 
                    "lado": last_result, 
                    "desc": f"Sequência de {current_streak} vitórias consecutivas",
                    "peso": min(3.0, current_streak*0.8)
                })
            current_streak = 1
            last_result = resultados[i]
    
    # Detecção de alternâncias
    alternancias = 0
    for i in range(1, n):
        if resultados[i] != resultados[i-1] and resultados[i] != 'T' and resultados[i-1] != 'T':
            alternancias += 1
    
    if alternancias/n >= 0.7:  # 70%+ de alternância
        padroes.append({
            "tipo": "ALTERNÂNCIA", 
            "lado": "AMBOS", 
            "desc": f"Padrão de alternância forte ({alternancias}/{n-1} trocas)",
            "peso": 2.0
        })
    
    # 4. Análise de Diferenças
    diferencas = abs(df_analise["Player"] - df_analise["Banker"])
    diff_media = diferencas.mean()
    
    if diff_media < 1.5:
        padroes.append({
            "tipo": "PROXIMIDADE", 
            "lado": "T", 
            "desc": f"Média de diferença baixa ({diff_media:.2f}) - favorece TIE",
            "peso": 1.8
        })
    
    # 5. Previsão com Modelo Simples
    if n > 10:
        try:
            X = df_analise[["Player", "Banker"]].values[:-1]
            y = df_analise["Resultado"].values[1:]
            
            # Converter para numérico
            y_num = [0 if r == 'P' else (1 if r == 'B' else 2) for r in y]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled, y_num)
            
            # Prever próximo resultado
            ultimo = scaler.transform([df_analise[["Player", "Banker"]].values[-1].reshape(1, -1)])
            pred = model.predict_proba(ultimo)[0]
            
            if max(pred) > 0.65:  # Confiança mínima de 65%
                lado_pred = ["P", "B", "T"][np.argmax(pred)]
                padroes.append({
                    "tipo": "PREVISÃO", 
                    "lado": lado_pred, 
                    "desc": f"Modelo preditivo sugere {lado_pred} (conf: {max(pred)*100:.1f}%)",
                    "peso": min(3.0, max(pred)*4)
                })
        except Exception as e:
            st.error(f"Erro no modelo preditivo: {str(e)}")
    
    return padroes

# 2. Sistema de Recomendação com Pontuação Hierárquica
def gerar_recomendacao(padroes):
    if not padroes:
        return "AGUARDAR", 10, "Sem padrões detectados. Aguarde mais dados.", "warning"
    
    # Agrupar padrões por lado
    scores = {"P": 0.0, "B": 0.0, "T": 0.0}
    detalhes = {"P": [], "B": [], "T": []}
    
    for padrao in padroes:
        lado = padrao["lado"]
        peso = padrao["peso"]
        
        if lado in scores:
            scores[lado] += peso
            detalhes[lado].append(padrao["desc"])
        elif lado == "AMBOS":
            scores["P"] += peso/2
            scores["B"] += peso/2
            detalhes["P"].append(padrao["desc"])
            detalhes["B"].append(padrao["desc"])
    
    # Calcular confiança
    total_score = sum(scores.values())
    confiancas = {lado: min(100, int(score/total_score * 100)) for lado, score in scores.items()}
    
    # Determinar recomendação
    max_lado = max(scores, key=scores.get)
    max_score = scores[max_lado]
    
    # Limiares de decisão
    if max_score > 4.0:
        acao = f"APOSTAR NO {'PLAYER' if max_lado == 'P' else 'BANKER' if max_lado == 'B' else 'TIE'}"
        tipo = "success"
        conf = confiancas[max_lado]
        detalhe = f"**Forte convergência de padrões** ({max_score:.1f} pontos):\n- " + "\n- ".join(detalhes[max_lado])
    elif max_score > 2.5:
        acao = f"CONSIDERAR {'PLAYER' if max_lado == 'P' else 'BANKER' if max_lado == 'B' else 'TIE'}"
        tipo = "warning"
        conf = confiancas[max_lado]
        detalhe = f"**Padrões moderados** ({max_score:.1f} pontos):\n- " + "\n- ".join(detalhes[max_lado])
    else:
        acao = "AGUARDAR"
        tipo = "warning"
        conf = 100 - max(confiancas.values())
        detalhe = "Padrões fracos ou conflitantes. Aguarde confirmação:\n- " + "\n- ".join(
            [f"{lado}: {score:.1f} pts" for lado, score in scores.items()])
    
    return acao, conf, detalhe, tipo

# --- Painel de Análise ---
st.markdown("---")
st.header("🧠 Análise Inteligente de Padrões")

# Analisar dados recentes (últimos 15-20 resultados)
n_analise = min(20, len(df))
df_recente = df.tail(n_analise).reset_index(drop=True)

# Detectar padrões
padroes = detectar_padroes_avancados(df_recente)
st.session_state.padroes_detectados = padroes

# Gerar recomendação
acao, confianca, detalhes, tipo = gerar_recomendacao(padroes)

# Exibir recomendação
st.markdown(f"""
<div class="stAlert alert-{tipo}">
    <div style="font-size: 1.5em;">{acao}</div>
    <div>Confiança: {confianca}%</div>
    <div style="font-size: 0.9em; margin-top: 10px;">{detalhes}</div>
</div>
""", unsafe_allow_html=True)

# --- Estatísticas Detalhadas ---
st.subheader("📈 Métricas Avançadas")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Jogos", len(df))
with col2:
    st.metric("Jogos Analisados", n_analise)
with col3:
    player_wins = len(df[df['Resultado'] == 'P'])
    st.metric("Vitórias Player", f"{player_wins} ({player_wins/len(df)*100:.1f}%)")
with col4:
    banker_wins = len(df[df['Resultado'] == 'B'])
    st.metric("Vitórias Banker", f"{banker_wins} ({banker_wins/len(df)*100:.1f}%)")

# --- Visualizações Gráficas ---
st.subheader("📊 Visualização de Padrões")

# Gráfico 1: Distribuição de Resultados
fig1 = px.pie(
    df, 
    names='Resultado', 
    title='Distribuição de Resultados',
    color='Resultado',
    color_discrete_map={'P': 'blue', 'B': 'red', 'T': 'green'}
)
fig1.update_traces(textposition='inside', textinfo='percent+label')

# Gráfico 2: Evolução Temporal
df['Indice'] = range(1, len(df)+1)
fig2 = px.line(
    df.tail(30), 
    x='Indice', 
    y=['Player', 'Banker'],
    title='Evolução das Somas (últimos 30 jogos)',
    markers=True
)
fig2.update_layout(yaxis_title="Soma", xaxis_title="Jogo")

# Gráfico 3: Heatmap de Frequência
freq_matrix = pd.crosstab(df['Player'], df['Banker'])
fig3 = px.imshow(
    freq_matrix,
    labels=dict(x="Banker", y="Player", color="Frequência"),
    title="Frequência Player vs Banker",
    aspect="auto"
)

# Exibir gráficos
col_graph1, col_graph2 = st.columns(2)
with col_graph1:
    st.plotly_chart(fig1, use_container_width=True)
with col_graph2:
    st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

# --- Detalhes dos Padrões Detectados ---
if padroes:
    st.subheader("🔍 Padrões Detectados")
    
    # Agrupar por tipo de padrão
    tipos = {}
    for padrao in padroes:
        if padrao['tipo'] not in tipos:
            tipos[padrao['tipo']] = []
        tipos[padrao['tipo']].append(padrao)
    
    # Exibir em abas
    tabs = st.tabs(list(tipos.keys()))
    
    for i, (tipo, padroes_tipo) in enumerate(tipos.items()):
        with tabs[i]:
            for padrao in padroes_tipo:
                st.markdown(f"""
                <div style="background: #2e2f3a; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <b>{padrao['lado']}</b> | Peso: {padrao['peso']:.1f}
                    <div style="margin-top: 5px;">{padrao['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

# --- Sistema de Alertas ---
st.subheader("🚨 Alertas Estratégicos")

# Verificar condições críticas
alertas = []
if len(df) > 10:
    # Alerta para sequências longas
    ultimo_resultado = df['Resultado'].iloc[-1]
    streak_count = 1
    for i in range(len(df)-2, -1, -1):
        if df['Resultado'].iloc[i] == ultimo_resultado:
            streak_count += 1
        else:
            break
    
    if streak_count >= 5:
        alertas.append(f"🚩 Sequência EXTREMA de {streak_count} vitórias consecutivas para {ultimo_resultado}")
    elif streak_count >= 4:
        alertas.append(f"⚠️ Sequência longa de {streak_count} vitórias consecutivas para {ultimo_resultado}")
    
    # Alerta para empates
    ultimo_tie_idx = df[df['Resultado'] == 'T'].index
    desde_ultimo_tie = len(df) - ultimo_tie_idx[-1] if len(ultimo_tie_idx) > 0 else len(df)
    
    if desde_ultimo_tie >= 15:
        alertas.append("🔥 Ciclo de TIE MADURO - Alta probabilidade de empate")
    elif desde_ultimo_tie >= 10:
        alertas.append("🔔 Ciclo de TIE APROXIMANDO - Fique atento")

# Exibir alertas
if alertas:
    for alerta in alertas:
        st.warning(alerta)
else:
    st.info("Nenhum alerta crítico detectado no momento")

# --- Painel de Controle ---
with st.expander("⚙️ Configurações Avançadas"):
    st.write("**Otimização de Parâmetros**")
    analise_range = st.slider("Número de jogos para análise", 5, 50, 20)
    limiar_confianca = st.slider("Limiar de confiança para apostas", 50, 90, 70)
    
    st.write("**Preferências de Estratégia**")
    estrategia = st.selectbox("Foco estratégico", [
        "Padrões de curto prazo", 
        "Tendências de longo prazo",
        "Detecção de empates",
        "Sequências de vitórias"
    ])
    
    if st.button("Atualizar Análise"):
        st.rerun()

st.markdown("---")
st.info("⚠️ **Aviso Importante**: Este sistema é uma ferramenta analítica. Jogos envolvem risco e resultados podem variar.")

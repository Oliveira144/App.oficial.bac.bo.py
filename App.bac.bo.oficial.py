import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter
from scipy import stats

# --- Configuração da Página ---
st.set_page_config(page_title="Bac Bo Inteligente", layout="wide", initial_sidebar_state="expanded")
st.title("🎲 Analisador Inteligente de Padrões - Bac Bo Evolution (v4.0 - Precisão Aprimorada)")

st.markdown("""
<style>
    .stApp {
        background-color: #262730;
        color: white;
    }
    .stTextInput>div>div>input {
        color: black;
    }
    .stTextArea>div>div>textarea {
        color: black;
    }
    /* Estilos para as caixas de Ação Recomendada */
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
    }
    .alert-success {
        background-color: #28a745; /* Verde */
        color: white;
    }
    .alert-danger {
        background-color: #dc3545; /* Vermelho */
        color: white;
    }
    .alert-warning {
        background-color: #ffc107; /* Amarelo */
        color: black;
    }
    .stSuccess {
        background-color: #28a745 !important;
        color: white !important;
    }
    .stWarning {
        background-color: #ffc107 !important;
        color: black !important;
    }
    .stError {
        background-color: #dc3545 !important;
        color: white !important;
    }
    /* Estilo para a tabela do histórico */
    .stDataFrame {
        color: black; /* Cor do texto dentro da tabela */
    }
    .stDataFrame thead th {
        background-color: #3e3f47; /* Cor de fundo do cabeçalho */
        color: white; /* Cor do texto do cabeçalho */
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f0f2f6; /* Cor de fundo para linhas pares */
    }
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: #ffffff; /* Cor de fundo para linhas ímpares */
    }
    .risk-bar {
        height: 20px;
        background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%);
        border-radius: 10px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
Olá! Bem-vindo ao analisador de padrões Bac Bo.
Para começar, insira os resultados recentes no campo abaixo.
Cada linha deve seguir o formato: **SomaPlayer,SomaBanker,Resultado**
(Ex: `11,4,P` para Player, `7,11,B` para Banker, `6,6,T` para Tie).
""")

# --- Inicialização do Session State para Histórico ---
if 'historico_dados' not in st.session_state:
    st.session_state.historico_dados = []

# --- Entrada de Dados Individualmente (para facilitar a adição) ---
st.subheader("Adicionar Novo Resultado")
col1, col2, col3 = st.columns(3)
with col1:
    player_soma = st.number_input("Soma Player (2-12)", min_value=2, max_value=12, value=7, key="player_soma_input")
with col2:
    banker_soma = st.number_input("Soma Banker (2-12)", min_value=2, max_value=12, value=7, key="banker_soma_input")
with col3:
    resultado_op = st.selectbox("Resultado", ['P', 'B', 'T'], key="resultado_select")

if st.button("Adicionar Linha ao Histórico"):
    st.session_state.historico_dados.append((player_soma, banker_soma, resultado_op))
    st.rerun()

# --- Exibir e Gerenciar Histórico ---
st.subheader("Histórico Atual")
if st.session_state.historico_dados:
    df_historico_exibicao = pd.DataFrame(st.session_state.historico_dados, columns=["Player", "Banker", "Resultado"])
    st.dataframe(df_historico_exibicao.tail(20), use_container_width=True)
    
    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        if st.button("Remover Última Linha"):
            if st.session_state.historico_dados:
                st.session_state.historico_dados.pop()
                st.rerun()
    with col_hist2:
        if st.button("Limpar Histórico Completo"):
            st.session_state.historico_dados = []
            st.rerun()
else:
    st.info("Nenhum dado no histórico ainda. Adicione resultados acima ou cole na caixa de texto.")

# --- Entrada de Dados em Massa ---
st.subheader("Adicionar Histórico em Massa (Cole aqui)")
historico_input_mass = st.text_area("Cole múltiplas linhas (Player,Banker,Resultado por linha)", height=150,
    value="")

if st.button("Processar Histórico em Massa"):
    linhas = historico_input_mass.strip().split("\n")
    novos_dados = []
    erros = []
    for linha in linhas:
        try:
            p_str, b_str, r = linha.strip().split(',')
            p = int(p_str)
            b = int(b_str)
            r = r.upper()

            if not (2 <= p <= 12 and 2 <= b <= 12):
                erros.append(f"Valores de soma inválidos na linha: {linha} (Soma deve ser entre 2 e 12).")
                continue
            if r not in ['P', 'B', 'T']:
                erros.append(f"Resultado inválido na linha: {linha} (Resultado deve ser 'P', 'B' ou 'T').")
                continue
            novos_dados.append((p, b, r))
        except ValueError:
            erros.append(f"Formato incorreto na linha: {linha} (Esperado: Player,Banker,Resultado).")
        except Exception as e:
            erros.append(f"Erro desconhecido na linha: {linha} - {e}")
    
    if erros:
        for erro in erros:
            st.error(erro)
    else:
        st.session_state.historico_dados.extend(novos_dados)
        st.success(f"{len(novos_dados)} linhas adicionadas com sucesso ao histórico!")
        st.rerun()

# --- Processar e Analisar Dados ---
if not st.session_state.historico_dados:
    st.warning("Adicione dados para iniciar a análise!")
    st.stop()

df = pd.DataFrame(st.session_state.historico_dados, columns=["Player", "Banker", "Resultado"])

# --- Algoritmos de Análise Avançados (Aprimorados) ---

def detectar_zigzag(resultado_series):
    zigzags = 0
    if len(resultado_series) < 3: return 0
    
    # Detecção mais eficiente usando diferenças
    diff = resultado_series.ne(resultado_series.shift())
    for i in range(2, len(diff)):
        if diff[i-1] and not diff[i] and diff[i-2]:
            zigzags += 1
    return zigzags

def detectar_alternancia_curta(resultado_series, n_ultimos=5):
    if len(resultado_series) < n_ultimos: return False

    alternancias = 0
    recentes = resultado_series.tail(n_ultimos).tolist()

    for i in range(len(recentes) - 1):
        if recentes[i] != recentes[i+1] and recentes[i+1] != 'T':
            alternancias += 1
    
    return alternancias / (n_ultimos - 1) >= 0.8

def detectar_streaks(df_analise):
    streaks = []
    if df_analise.empty: return streaks

    current_streak_data = {
        'lado': df_analise["Resultado"].iloc[0],
        'contagem': 1,
        'somas': [df_analise["Player"].iloc[0] if df_analise["Resultado"].iloc[0] == 'P' else df_analise["Banker"].iloc[0]],
        'forca': 0  # Nova métrica de força
    }

    for i in range(1, len(df_analise)):
        if df_analise["Resultado"].iloc[i] == current_streak_data['lado']:
            current_streak_data['contagem'] += 1
            soma_atual = df_analise["Player"].iloc[i] if current_streak_data['lado'] == 'P' else df_analise["Banker"].iloc[i]
            current_streak_data['somas'].append(soma_atual)
            
            # Cálculo da força do streak (média das somas)
            current_streak_data['forca'] = sum(current_streak_data['somas']) / len(current_streak_data['somas'])
        else:
            if current_streak_data['contagem'] >= 2:
                streaks.append(current_streak_data)
            current_streak_data = {
                'lado': df_analise["Resultado"].iloc[i],
                'contagem': 1,
                'somas': [df_analise["Player"].iloc[i] if df_analise["Resultado"].iloc[i] == 'P' else df_analise["Banker"].iloc[i]],
                'forca': 0
            }
    
    if current_streak_data['contagem'] >= 2:
        streaks.append(current_streak_data)
    return streaks

def freq_resultados(df_analise):
    total = len(df_analise)
    if total == 0: return {'P': 0, 'B': 0, 'T': 0}
    freq = df_analise["Resultado"].value_counts(normalize=True).reindex(['P', 'B', 'T'], fill_value=0) * 100
    return freq.to_dict()

def analisar_somas_proximas(df_analise, n_ultimos=7):
    somas_proximas_detectadas = []
    if len(df_analise) < 2: return somas_proximas_detectadas

    df_recentes = df_analise.tail(n_ultimos).reset_index(drop=True)

    for i in range(len(df_recentes) - 1):
        r_atual = df_recentes["Resultado"].iloc[i]
        r_prox = df_recentes["Resultado"].iloc[i+1]

        if r_atual == r_prox and r_atual != 'T':
            soma_atual = df_recentes["Player"].iloc[i] if r_atual == 'P' else df_recentes["Banker"].iloc[i]
            soma_prox = df_recentes["Player"].iloc[i+1] if r_prox == 'P' else df_recentes["Banker"].iloc[i+1]

            if abs(soma_atual - soma_prox) <= 1 or ((soma_atual >= 7 and soma_atual <= 8) and (soma_prox >= 7 and soma_prox <= 8)):
                somas_proximas_detectadas.append({
                    'lado': r_atual,
                    'soma_1': soma_atual,
                    'soma_2': soma_prox
                })
    
    # Detecção para sequências de 3+ vitórias com somas próximas
    if len(df_recentes) >= 3:
        for i in range(len(df_recentes) - 2):
            if (df_recentes["Resultado"].iloc[i] == df_recentes["Resultado"].iloc[i+1] == 
                df_recentes["Resultado"].iloc[i+2] and df_recentes["Resultado"].iloc[i] != 'T'):
                
                lado = df_recentes["Resultado"].iloc[i]
                somas = [
                    df_recentes["Player"].iloc[i] if lado == 'P' else df_recentes["Banker"].iloc[i],
                    df_recentes["Player"].iloc[i+1] if lado == 'P' else df_recentes["Banker"].iloc[i+1],
                    df_recentes["Player"].iloc[i+2] if lado == 'P' else df_recentes["Banker"].iloc[i+2]
                ]
                
                # Verificar se as somas estão em intervalo próximo
                if max(somas) - min(somas) <= 2:
                    somas_proximas_detectadas.append({
                        'lado': lado,
                        'somas': somas,
                        'tamanho': 3
                    })
    
    return somas_proximas_detectadas

def analisar_near_misses_tie(df_analise, n_ultimos=10):
    near_misses = 0
    if len(df_analise) < n_ultimos: return 0
    
    recentes = df_analise.tail(n_ultimos)
    for index, row in recentes.iterrows():
        if row['Resultado'] != 'T':
            if abs(row['Player'] - row['Banker']) == 1:
                near_misses += 1
    return near_misses

def analisar_soma_vencedora_recorrente(df_analise, n_ultimos=15):
    if len(df_analise) < n_ultimos: return None, 0

    vencedoras = []
    for index, row in df_analise.tail(n_ultimos).iterrows():
        if row['Resultado'] == 'P':
            vencedoras.append(row['Player'])
        elif row['Resultado'] == 'B':
            vencedoras.append(row['Banker'])
    
    if not vencedoras: return None, 0

    contagem_somas = Counter(vencedoras)
    soma_mais_comum = None
    max_ocorrencias = 0

    for soma, count in contagem_somas.items():
        if count > max_ocorrencias:
            max_ocorrencias = count
            soma_mais_comum = soma
    
    if len(vencedoras) > 0:
        return soma_mais_comum, (max_ocorrencias / len(vencedoras)) * 100
    return None, 0

def analisar_distribuicao_somas(df_analise, n_ultimos=20):
    if len(df_analise) < n_ultimos:
        return {'P_Altas_Pct': 0, 'B_Altas_Pct': 0}

    df_recentes = df_analise.tail(n_ultimos)

    player_somas_altas = df_recentes[df_recentes['Resultado'] == 'P']['Player'].apply(lambda x: 1 if x >= 8 and x <= 12 else 0).sum()
    total_player_wins = (df_recentes['Resultado'] == 'P').sum()

    banker_somas_altas = df_recentes[df_recentes['Resultado'] == 'B']['Banker'].apply(lambda x: 1 if x >= 8 and x <= 12 else 0).sum()
    total_banker_wins = (df_recentes['Resultado'] == 'B').sum()

    return {
        'P_Altas_Pct': (player_somas_altas / total_player_wins * 100) if total_player_wins > 0 else 0,
        'B_Altas_Pct': (banker_somas_altas / total_banker_wins * 100) if total_banker_wins > 0 else 0,
    }

def detectar_ciclos_hibridos(df_analise, janela=10):
    """Detecta padrões complexos como P-P-B-B, P-B-P-B, etc."""
    if len(df_analise) < janela:
        return None, 0
    
    padroes = {
        'ziguezague': 0,
        'blocos': 0,
        'tie_intermitente': 0
    }
    
    # Implementação simplificada para demonstração
    resultados = df_analise['Resultado'].tail(janela).tolist()
    
    # Detecção de ziguezague (alternância constante)
    alternancias = 0
    for i in range(1, len(resultados)):
        if resultados[i] != resultados[i-1] and resultados[i] != 'T' and resultados[i-1] != 'T':
            alternancias += 1
    
    if alternancias / (len(resultados) - 1) > 0.75:
        padroes['ziguezague'] = alternancias
    
    # Detecção de blocos (sequências de 2+ do mesmo lado)
    blocos = 0
    current_block = 1
    for i in range(1, len(resultados)):
        if resultados[i] == resultados[i-1] and resultados[i] != 'T':
            current_block += 1
        else:
            if current_block >= 2:
                blocos += 1
            current_block = 1
    
    if current_block >= 2:
        blocos += 1
    
    padroes['blocos'] = blocos
    
    # Detecção de Tie intermitente
    ties = resultados.count('T')
    if ties > 0 and ties < len(resultados) * 0.5:
        padroes['tie_intermitente'] = ties
    
    # Encontrar padrão mais forte
    padrao_detectado = max(padroes, key=padroes.get)
    forca = padroes[padrao_detectado] / janela
    
    return padrao_detectado, forca

def detectar_mudanca_regime(df_analise):
    """Detecta mudanças significativas nos padrões do jogo"""
    if len(df_analise) < 30:
        return False
    
    # Dividir em segmentos históricos
    segmento_antigo = df_analise.iloc[:15]
    segmento_recente = df_analise.iloc[-15:]
    
    # Calcular diferenças chave
    freq_antigo = freq_resultados(segmento_antigo)
    freq_recente = freq_resultados(segmento_recente)
    
    diff_freq_p = abs(freq_antigo.get('P', 0) - freq_recente.get('P', 0))
    diff_freq_b = abs(freq_antigo.get('B', 0) - freq_recente.get('B', 0))
    diff_freq_t = abs(freq_antigo.get('T', 0) - freq_recente.get('T', 0))
    
    diff_media_player = abs(segmento_antigo['Player'].mean() - segmento_recente['Player'].mean())
    diff_media_banker = abs(segmento_antigo['Banker'].mean() - segmento_recente['Banker'].mean())
    
    # Verificar se houve mudança significativa
    return (diff_freq_p > 20 or diff_freq_b > 20 or diff_freq_t > 10 or 
            diff_media_player > 1.5 or diff_media_banker > 1.5)

def detectar_conflitos(df_analise):
    """Detecta conflitos entre padrões detectados"""
    if len(df_analise) < 20:
        return 0
    
    # Verificar divergência entre frequência de resultados e média de somas
    freq = freq_resultados(df_analise)
    media_player = df_analise['Player'].mean()
    media_banker = df_analise['Banker'].mean()
    
    conflitos = 0
    
    # Se Player tem maior frequência mas soma média menor
    if freq['P'] > freq['B'] and media_player < media_banker:
        conflitos += 1
    
    # Se Banker tem maior frequência mas soma média menor
    if freq['B'] > freq['P'] and media_banker < media_player:
        conflitos += 1
    
    # Se há muitos near misses mas poucos ties
    near_misses = analisar_near_misses_tie(df_analise)
    ties = (df_analise['Resultado'] == 'T').sum()
    if near_misses > 5 and ties < 2:
        conflitos += 1
    
    return conflitos

def gerar_relatorio_sugestoes(df_completo):
    n_jogos_sugestao = 20
    df_analise_sugestao = df_completo.tail(n_jogos_sugestao).copy()

    sugestoes_geradas = []
    logicas_ativas = {
        'tie_ciclo_forte': False,
        'tie_ciclo_medio': False,
        'player_vantagem_soma': False,
        'banker_vantagem_soma': False,
        'zigzag_soma_real': False,
        'zigzag_fraco_apenas_cor': False,
        'alternancia_curta_ativa': False,
        'streak_player_verdadeiro': False,
        'streak_banker_verdadeiro': False,
        'streak_player_normal': False,
        'streak_banker_normal': False,
        'somas_proximas_player': False,
        'somas_proximas_banker': False,
        'near_misses_tie_ativos': False,
        'soma_vencedora_player_recorrente': False,
        'soma_vencedora_banker_recorrente': False,
        'distribuicao_somas_player_alta': False,
        'distribuicao_somas_banker_alta': False,
        'mudanca_regime': False,
        'conflito_padroes': False
    }

    if df_analise_sugestao.empty:
        return sugestoes_geradas, logicas_ativas

    # 1. Empates e Ciclo de TIE
    freq_t = freq_resultados(df_analise_sugestao).get('T', 0)
    ultimos_ties_idx = df_completo[df_completo["Resultado"] == "T"].index
    desde_ultimo_tie = len(df_completo)

    if not ultimos_ties_idx.empty:
        desde_ultimo_tie = len(df_completo) - ultimos_ties_idx[-1]

    if (6 <= desde_ultimo_tie <= 10): 
        sugestoes_geradas.append(f"🟢 **TIE (Ciclo Madura)**: Padrão estatístico (6-10 jogos sem TIE) sugere TIE próximo. Passaram {desde_ultimo_tie} rodadas desde o último TIE.")
        logicas_ativas['tie_ciclo_forte'] = True
    elif (11 <= desde_ultimo_tie <= 15) or (freq_t < 15 and desde_ultimo_tie > 8):
        sugestoes_geradas.append(f"🟢 **TIE (Ciclo Médio)**: Padrão cíclico (11-15 jogos ou baixa freq). Passaram {desde_ultimo_tie} rodadas desde o último TIE.")
        logicas_ativas['tie_ciclo_medio'] = True
    elif freq_t < 10 and desde_ultimo_tie > 15:
        sugestoes_geradas.append(f"🟢 **TIE (Ciclo Atrasado)**: Muito tempo sem TIE (>15 jogos) e frequência baixa. Aumenta a chance. Passaram {desde_ultimo_tie} rodadas desde o último TIE.")
        logicas_ativas['tie_ciclo_forte'] = True
    
    # Near Misses para TIE
    near_misses_count = analisar_near_misses_tie(df_analise_sugestao, n_ultimos=10)
    if near_misses_count >= 3:
        sugestoes_geradas.append(f"🟡 **TIE (Near Misses)**: {near_misses_count} 'quase empates' recentes. Aumenta a chance de um TIE real.")
        logicas_ativas['near_misses_tie_ativos'] = True

    # 2. Vantagem de Soma
    player_med = df_analise_sugestao["Player"].mean()
    banker_med = df_analise_sugestao["Banker"].mean()
    
    if player_med > banker_med + 1.5:
        sugestoes_geradas.append(f"🔵 **PLAYER**: Soma média ({player_med:.1f}) consistentemente mais alta. Vantagem no lance de dados.")
        logicas_ativas['player_vantagem_soma'] = True
    elif banker_med > player_med + 1.5:
        sugestoes_geradas.append(f"🔴 **BANKER**: Soma média ({banker_med:.1f}) consistentemente mais alta. Vantagem no lance de dados.")
        logicas_ativas['banker_vantagem_soma'] = True
    
    # 3. ZigZag e Alternância Curta
    zigzag_count = detectar_zigzag(df_analise_sugestao["Resultado"])
    alternancia_curta_ativa = detectar_alternancia_curta(df_analise_sugestao["Resultado"], n_ultimos=5)
    
    is_zigzag_soma_real = False
    if zigzag_count >= 2 and len(df_analise_sugestao) >= 5:
        df_pb = df_analise_sugestao[(df_analise_sugestao['Resultado'] == 'P') | (df_analise_sugestao['Resultado'] == 'B')]
        if len(df_pb) >= 4:
            somas_p_zigzag = df_pb[df_pb['Resultado'] == 'P']['Player'].mean()
            somas_b_zigzag = df_pb[df_pb['Resultado'] == 'B']['Banker'].mean()
            if abs(somas_p_zigzag - somas_b_zigzag) <= 1.0:
                is_zigzag_soma_real = True

    if is_zigzag_soma_real:
        ultimo_resultado = df_analise_sugestao["Resultado"].iloc[-1]
        sugestoes_geradas.append(f"🔁 **{('BANKER' if ultimo_resultado == 'P' else 'PLAYER')} (ZigZag Real por Soma)**: Padrão ZigZag ativo com somas próximas. Sugere alternância baseada em ritmo dos dados.")
        logicas_ativas['zigzag_soma_real'] = True
    elif alternancia_curta_at

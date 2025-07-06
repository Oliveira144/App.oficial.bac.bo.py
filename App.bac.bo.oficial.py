import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter

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
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
    }
    .alert-success {
        background-color: #28a745;
        color: white;
    }
    .alert-danger {
        background-color: #dc3545;
        color: white;
    }
    .alert-warning {
        background-color: #ffc107;
        color: black;
    }
    .stDataFrame {
        color: black;
    }
    .stDataFrame thead th {
        background-color: #3e3f47;
        color: white;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f0f2f6;
    }
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: #ffffff;
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

# --- Entrada de Dados Individualmente ---
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
historico_input_mass = st.text_area("Cole múltiplas linhas (Player,Banker,Resultado por linha)", height=150, value="")

if st.button("Processar Histórico em Massa"):
    linhas = historico_input_mass.strip().split("\n")
    novos_dados = []
    erros = []
    for linha in linhas:
        try:
            partes = linha.strip().split(',')
            if len(partes) < 3:
                erros.append(f"Formato incorreto na linha: {linha} (Esperado: Player,Banker,Resultado).")
                continue
                
            p_str, b_str, r = partes
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

# --- Algoritmos de Análise Avançados ---
def detectar_zigzag(resultado_series):
    zigzags = 0
    if len(resultado_series) < 3: 
        return 0
    
    for i in range(2, len(resultado_series)):
        if (resultado_series.iloc[i-2] == resultado_series.iloc[i] and 
            resultado_series.iloc[i-2] != resultado_series.iloc[i-1]):
            zigzags += 1
    return zigzags

def detectar_alternancia_curta(resultado_series, n_ultimos=5):
    if len(resultado_series) < n_ultimos: 
        return False

    alternancias = 0
    recentes = resultado_series.tail(n_ultimos).tolist()

    for i in range(len(recentes) - 1):
        if recentes[i] != recentes[i+1] and recentes[i+1] != 'T':
            alternancias += 1
    
    return alternancias / (n_ultimos - 1) >= 0.8

def detectar_streaks(df_analise):
    streaks = []
    if df_analise.empty: 
        return streaks

    current_streak_data = {
        'lado': df_analise["Resultado"].iloc[0],
        'contagem': 1,
        'somas': [df_analise["Player"].iloc[0] if df_analise["Resultado"].iloc[0] == 'P' else df_analise["Banker"].iloc[0]],
        'forca': 0
    }

    for i in range(1, len(df_analise)):
        if df_analise["Resultado"].iloc[i] == current_streak_data['lado']:
            current_streak_data['contagem'] += 1
            soma_atual = df_analise["Player"].iloc[i] if current_streak_data['lado'] == 'P' else df_analise["Banker"].iloc[i]
            current_streak_data['somas'].append(soma_atual)
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
    if total == 0: 
        return {'P': 0, 'B': 0, 'T': 0}
    freq = df_analise["Resultado"].value_counts(normalize=True).reindex(['P', 'B', 'T'], fill_value=0) * 100
    return freq.to_dict()

def analisar_somas_proximas(df_analise, n_ultimos=7):
    somas_proximas_detectadas = []
    if len(df_analise) < 2: 
        return somas_proximas_detectadas

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
    
    # Detecção para sequências de 3+ vitórias
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
                
                if max(somas) - min(somas) <= 2:
                    somas_proximas_detectadas.append({
                        'lado': lado,
                        'somas': somas,
                        'tamanho': 3
                    })
    
    return somas_proximas_detectadas

def analisar_near_misses_tie(df_analise, n_ultimos=10):
    near_misses = 0
    if len(df_analise) < n_ultimos: 
        return 0
    
    recentes = df_analise.tail(n_ultimos)
    for index, row in recentes.iterrows():
        if row['Resultado'] != 'T':
            if abs(row['Player'] - row['Banker']) == 1:
                near_misses += 1
    return near_misses

def analisar_soma_vencedora_recorrente(df_analise, n_ultimos=15):
    if len(df_analise) < n_ultimos: 
        return None, 0

    vencedoras = []
    for index, row in df_analise.tail(n_ultimos).iterrows():
        if row['Resultado'] == 'P':
            vencedoras.append(row['Player'])
        elif row['Resultado'] == 'B':
            vencedoras.append(row['Banker'])
    
    if not vencedoras: 
        return None, 0

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
    if len(df_analise) < janela:
        return None, 0
    
    padroes = {
        'ziguezague': 0,
        'blocos': 0,
        'tie_intermitente': 0
    }
    
    resultados = df_analise['Resultado'].tail(janela).tolist()
    
    # Detecção de ziguezague
    alternancias = 0
    for i in range(1, len(resultados)):
        if resultados[i] != resultados[i-1] and resultados[i] != 'T' and resultados[i-1] != 'T':
            alternancias += 1
    
    if alternancias / (len(resultados) - 1) > 0.75:
        padroes['ziguezague'] = alternancias
    
    # Detecção de blocos
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
    
    padrao_detectado = max(padroes, key=padroes.get)
    forca = padroes[padrao_detectado] / janela
    
    return padrao_detectado, forca

def detectar_mudanca_regime(df_analise):
    if len(df_analise) < 30:
        return False
    
    segmento_antigo = df_analise.iloc[:15]
    segmento_recente = df_analise.iloc[-15:]
    
    freq_antigo = freq_resultados(segmento_antigo)
    freq_recente = freq_resultados(segmento_recente)
    
    diff_freq_p = abs(freq_antigo.get('P', 0) - freq_recente.get('P', 0))
    diff_freq_b = abs(freq_antigo.get('B', 0) - freq_recente.get('B', 0))
    diff_freq_t = abs(freq_antigo.get('T', 0) - freq_recente.get('T', 0))
    
    diff_media_player = abs(segmento_antigo['Player'].mean() - segmento_recente['Player'].mean())
    diff_media_banker = abs(segmento_antigo['Banker'].mean() - segmento_recente['Banker'].mean())
    
    return (diff_freq_p > 20 or diff_freq_b > 20 or diff_freq_t > 10 or 
            diff_media_player > 1.5 or diff_media_banker > 1.5)

def detectar_conflitos(df_analise):
    if len(df_analise) < 20:
        return 0
    
    freq = freq_resultados(df_analise)
    media_player = df_analise['Player'].mean()
    media_banker = df_analise['Banker'].mean()
    
    conflitos = 0
    
    if freq['P'] > freq['B'] and media_player < media_banker:
        conflitos += 1
    
    if freq['B'] > freq['P'] and media_banker < media_player:
        conflitos += 1
    
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
    
    # 3. ZigZag e Alternância Curta (CORREÇÃO APLICADA AQUI)
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
    elif alternancia_curta_ativa:  # CORREÇÃO: linha completa com ":" no final
        sugestoes_geradas.append("🔄 **Alternância Curta Ativa (Sem Padrão Forte)**: O jogo está alternando P/B frequentemente, mas sem um padrão de soma ou ritmo claro.")
        logicas_ativas['alternancia_curta_ativa'] = True
    else:
        sugestoes_geradas.append("⚠️ **ZigZag Fraco / Sem Padrão de Alternância Clara**: Não há ZigZag real ou alternância consistente. Não se baseie apenas na cor.")
        logicas_ativas['zigzag_fraco_apenas_cor'] = True

    # 4. Streaks
    streaks = detectar_streaks(df_analise_sugestao)
    if streaks:
        ultimo_streak = streaks[-1]
        lado_nome = ultimo_streak['lado']
        somas_altas_no_streak = [s for s in ultimo_streak['somas'] if s >= 8 and s <= 12]
        
        if ultimo_streak['contagem'] >= 3:
            if len(somas_altas_no_streak) >= (ultimo_streak['contagem'] * 0.75):
                sugestoes_geradas.append(f"🔥 **{lado_nome} (Streak VERDADEIRO c/ Somas Altas)**: {ultimo_streak['contagem']} consecutivos com somas consistentemente altas. Forte tendência de seguir o lado.")
                if lado_nome == 'P': 
                    logicas_ativas['streak_player_verdadeiro'] = True
                elif lado_nome == 'B': 
                    logicas_ativas['streak_banker_verdadeiro'] = True
            else:
                sugestoes_geradas.append(f"🔥 **{lado_nome} (Streak de {ultimo_streak['contagem']} )**: Tendência de seguir, mas as somas não foram predominantemente altas.")
                if lado_nome == 'P': 
                    logicas_ativas['streak_player_normal'] = True
                elif lado_nome == 'B': 
                    logicas_ativas['streak_banker_normal'] = True
    
    # 5. Repetição de Somas Próximas
    somas_prox = analisar_somas_proximas(df_analise_sugestao, n_ultimos=7)
    if somas_prox:
        for sp in somas_prox:
            if 'tamanho' in sp and sp['tamanho'] == 3:
                msg = f"✨ **Padrão de Somas Próximas em {sp['lado']}**: 3 vitórias consecutivas com somas {sp['somas']}. Sugere forte repetição."
            else:
                msg = f"✨ **Padrão de Somas Próximas em {sp['lado']}**: Duas vitórias com somas {sp['soma_1']} e {sp['soma_2']}. Sugere repetição."
                
            if msg not in sugestoes_geradas:
                sugestoes_geradas.append(msg)
                if sp['lado'] == 'P': 
                    logicas_ativas['somas_proximas_player'] = True
                elif sp['lado'] == 'B': 
                    logicas_ativas['somas_proximas_banker'] = True

    # 6. Soma Vencedora Recorrente
    soma_comum, pct_comum = analisar_soma_vencedora_recorrente(df_analise_sugestao, n_ultimos=15)
    if soma_comum is not None and pct_comum >= 40:
        player_vitorias_com_soma = df_analise_sugestao[(df_analise_sugestao['Resultado'] == 'P') & (df_analise_sugestao['Player'] == soma_comum)].shape[0]
        banker_vitorias_com_soma = df_analise_sugestao[(df_analise_sugestao['Resultado'] == 'B') & (df_analise_sugestao['Banker'] == soma_comum)].shape[0]

        if player_vitorias_com_soma > banker_vitorias_com_soma:
            sugestoes_geradas.append(f"📊 **PLAYER (Soma Recorrente)**: Soma {soma_comum} aparece frequentemente em vitórias de Player ({pct_comum:.1f}% das vitórias).")
            logicas_ativas['soma_vencedora_player_recorrente'] = True
        elif banker_vitorias_com_soma > player_vitorias_com_soma:
            sugestoes_geradas.append(f"📊 **BANKER (Soma Recorrente)**: Soma {soma_comum} aparece frequentemente em vitórias de Banker ({pct_comum:.1f}% das vitórias).")
            logicas_ativas['soma_vencedora_banker_recorrente'] = True

    # 7. Distribuição de Somas (Altas)
    dist_somas = analisar_distribuicao_somas(df_analise_sugestao, n_ultimos=20)
    if dist_somas['P_Altas_Pct'] >= 60:
        sugestoes_geradas.append(f"📈 **PLAYER (Somas Altas)**: Player vencendo com somas altas em {dist_somas['P_Altas_Pct']:.1f}% das vezes (8 a 12).")
        logicas_ativas['distribuicao_somas_player_alta'] = True
    if dist_somas['B_Altas_Pct'] >= 60:
        sugestoes_geradas.append(f"📈 **BANKER (Somas Altas)**: Banker vencendo com somas altas em {dist_somas['B_Altas_Pct']:.1f}% das vezes (8 a 12).")
        logicas_ativas['distribuicao_somas_banker_alta'] = True
    
    # 8. Detecção de mudança de regime
    if detectar_mudanca_regime(df_completo):
        sugestoes_geradas.append("⚠️ **MUDANÇA DE REGIME DETECTADA**: Padrões históricos recentes diferem significativamente. Cautela extra recomendada.")
        logicas_ativas['mudanca_regime'] = True
    
    # 9. Detecção de conflitos
    conflitos = detectar_conflitos(df_completo)
    if conflitos > 0:
        sugestoes_geradas.append(f"⚠️ **CONFLITO DE PADRÕES DETECTADO**: {conflitos} indicadores contraditórios encontrados. Análise cuidadosa necessária.")
        logicas_ativas['conflito_padroes'] = True
    
    # 10. Padrões híbridos
    padrao_hibrido, forca_hibrido = detectar_ciclos_hibridos(df_analise_sugestao)
    if forca_hibrido > 0.6:
        sugestoes_geradas.append(f"🔄 **Padrão Híbrido Detectado ({padrao_hibrido.upper()})**: Ciclo complexo com força {forca_hibrido:.2f}. Considere estratégia adaptativa.")
        logicas_ativas[f'padrao_hibrido_{padrao_hibrido}'] = True
    
    return sugestoes_geradas, logicas_ativas

def calcular_confianca(df_analise, score, logicas_ativas):
    confianca_base = min(100, max(score.values()) * 15)
    penalidade_dados = max(0, (50 - len(df_analise)) * 0.5) if len(df_analise) < 50 else 0
    bônus_convergencia = 10 if sum(logicas_ativas.values()) > 3 else 0
    penalidade_conflito = 15 if (score['P'] > 3 and score['B'] > 3) else 0
    penalidade_regime = 20 if logicas_ativas.get('mudanca_regime', False) else 0
    
    confianca_final = max(10, min(100, 
        confianca_base - penalidade_dados + bônus_convergencia - penalidade_conflito - penalidade_regime
    ))
    
    return int(confianca_final)

def determinar_acao_recomendada_inteligente(logicas_ativas, sugestoes_detalhadas):
    pesos = {
        'tie_ciclo_forte': 4.0,
        'tie_ciclo_medio': 2.0,
        'near_misses_tie_ativos': 1.5,
        'streak_verdadeiro': lambda f: min(6.0, f * 1.5) if f > 0 else 0,
        'streak_normal': 1.0,
        'vantagem_soma': lambda d: min(3.0, d * 0.5),
        'somas_proximas': lambda t: min(3.0, t * 0.8),
        'soma_vencedora_recorrente': 1.8,
        'distribuicao_somas_alta': 1.2,
        'zigzag_soma_real': 1.5,
        'alternancia_curta_ativa': 0.5,
        'padrao_hibrido_ziguezague': 2.0,
        'padrao_hibrido_blocos': 2.5,
        'mudanca_regime': -3.0
    }

    score = {'P': 0.0, 'B': 0.0, 'T': 0.0}
    justificativas_finais = []
    aposta_player_forte = False
    aposta_banker_forte = False
    aposta_tie_forte = False

    # Avaliação do TIE
    if logicas_ativas['tie_ciclo_forte']:
        score['T'] += pesos['tie_ciclo_forte']
        justificativas_finais.append("Padrão de Empate (TIE) FORTEMENTE maduro/atrasado.")
        aposta_tie_forte = True
    elif logicas_ativas['tie_ciclo_medio']:
        score['T'] += pesos['tie_ciclo_medio']
        justificativas_finais.append("Padrão de Empate (TIE) em ciclo médio.")
        aposta_tie_forte = True
    if logicas_ativas['near_misses_tie_ativos']:
        score['T'] += pesos['near_misses_tie_ativos']
        justificativas_finais.append("Vários 'quase empates' (Near Misses) recentes.")

    # Avaliação de Player
    if logicas_ativas['streak_player_verdadeiro']:
        streaks = detectar_streaks(df.tail(20))
        streak_forca = streaks[-1]['forca'] if streaks else 0
        score['P'] += pesos['streak_verdadeiro'](streak_forca)
        justificativas_finais.append(f"Streak VERDADEIRO de PLAYER com força {streak_forca:.1f}.")
        aposta_player_forte = True
        
    if logicas_ativas['player_vantagem_soma']:
        diff = df['Player'].mean() - df['Banker'].mean()
        score['P'] += pesos['vantagem_soma'](diff)
        justificativas_finais.append(f"PLAYER com vantagem de soma ({diff:.1f}).")
        if logicas_ativas['somas_proximas_player'] or logicas_ativas['distribuicao_somas_player_alta']:
            aposta_player_forte = True
            
    if logicas_ativas['somas_proximas_player']:
        seq_len = max([sp['tamanho'] for sp in analisar_somas_proximas(df.tail(10)) if 'tamanho' in sp and sp['lado'] == 'P'], default=2)
        score['P'] += pesos['somas_proximas'](seq_len)
        justificativas_finais.append(f"Sequência de {seq_len} vitórias PLAYER com somas próximas.")
        
    if logicas_ativas['soma_vencedora_player_recorrente']:
        score['P'] += pesos['soma_vencedora_recorrente']
        justificativas_finais.append("Soma vencedora recorrente para PLAYER.")
        
    if logicas_ativas['distribuicao_somas_player_alta']:
        score['P'] += pesos['distribuicao_somas_alta']
        justificativas_finais.append("PLAYER vencendo com somas predominantemente altas.")
    
    # Avaliação de Banker
    if logicas_ativas['streak_banker_verdadeiro']:
        streaks = detectar_streaks(df.tail(20))
        streak_forca = streaks[-1]['forca'] if streaks else 0
        score['B'] += pesos['streak_verdadeiro'](streak_forca)
        justificativas_finais.append(f"Streak VERDADEIRO de BANKER com força {streak_forca:.1f}.")
        aposta_banker_forte = True
        
    if logicas_ativas['banker_vantagem_soma']:
        diff = df['Banker'].mean() - df['Player'].mean()
        score['B'] += pesos['vantagem_soma'](diff)
        justificativas_finais.append(f"BANKER com vantagem de soma ({diff:.1f}).")
        if logicas_ativas['somas_proximas_banker'] or logicas_ativas['distribuicao_somas_banker_alta']:
            aposta_banker_forte = True
            
    if logicas_ativas['somas_proximas_banker']:
        seq_len = max([sp['tamanho'] for sp in analisar_somas_proximas(df.tail(10)) if 'tamanho' in sp and sp['lado'] == 'B'], default=2)
        score['B'] += pesos['somas_proximas'](seq_len)
        justificativas_finais.append(f"Sequência de {seq_len} vitórias BANKER com somas próximas.")
        
    if logicas_ativas['soma_vencedora_banker_recorrente']:
        score['B'] += pesos['soma_vencedora_recorrente']
        justificativas_finais.append("Soma vencedora recorrente para BANKER.")
        
    if logicas_ativas['distribuicao_somas_banker_alta']:
        score['B'] += pesos['distribuicao_somas_alta']
        justificativas_finais.append("BANKER vencendo com somas predominantemente altas.")

    # ZigZag Real por Soma
    if logicas_ativas['zigzag_soma_real']:
        if not df.empty and len(df) >= 1:
            ultimo_resultado = df.tail(1)["Resultado"].iloc[0]
            if ultimo_resultado == 'P':
                score['B'] += pesos['zigzag_soma_real']
                justificativas_finais.append("Padrão ZigZag REAL por soma ativo (sugere BANKER).")
                aposta_banker_forte = True
            elif ultimo_resultado == 'B':
                score['P'] += pesos['zigzag_soma_real']
                justificativas_finais.append("Padrão ZigZag REAL por soma ativo (sugere PLAYER).")
                aposta_player_forte = True
    
    # Padrões híbridos
    if logicas_ativas.get('padrao_hibrido_ziguezague', False):
        score['P'] += pesos['padrao_hibrido_ziguezague'] * 0.5
        score['B'] += pesos['padrao_hibrido_ziguezague'] * 0.5
        justificativas_finais.append("Padrão híbrido de ziguezague detectado.")
        
    if logicas_ativas.get('padrao_hibrido_blocos', False):
        if not df.empty:
            ultimo_resultado = df.tail(1)["Resultado"].iloc[0]
            if ultimo_resultado == 'P':
                score['P'] += pesos['padrao_hibrido_blocos']
            elif ultimo_resultado == 'B':
                score['B'] += pesos['padrao_hibrido_blocos']
        justificativas_finais.append("Padrão híbrido de blocos detectado.")

    # Penalidade por mudança de regime
    if logicas_ativas.get('mudanca_regime', False):
        score['P'] += pesos['mudanca_regime']
        score['B'] += pesos['mudanca_regime']
        score['T'] += pesos['mudanca_regime']
        justificativas_finais.append("Penalidade aplicada por mudança de regime detectada.")

    # --- REGRAS DE DECISÃO FINAL ---
    if logicas_ativas['zigzag_fraco_apenas_cor'] and \
       not (logicas_ativas['streak_player_verdadeiro'] or logicas_ativas['streak_banker_verdadeiro'] or logicas_ativas['zigzag_soma_real']):
        return "AGUARDAR", 20, "**Cenário instável/fraco**: ZigZag fraco e sem Streaks Verdadeiros ou ZigZag Real. Não se baseie apenas na cor. " + ("Detalhes: " + ", ".join(justificativas_finais)), "warning"

    if ((logicas_ativas['player_vantagem_soma'] and not (logicas_ativas['streak_player_verdadeiro'] or logicas_ativas['zigzag_soma_real'])) or
        (logicas_ativas['banker_vantagem_soma'] and not (logicas_ativas['streak_banker_verdadeiro'] or logicas_ativas['zigzag_soma_real']))):
        if not aposta_tie_forte:
             return "AGUARDAR", 40, "**Cenário incerto**: Vantagem de soma sem 'streak claro' ou ZigZag Real. " + ("Detalhes: " + ", ".join(justificativas_finais)), "warning"

    limiar_conflito = 3.0
    diferenca_minima_para_dominancia = 2.0
    
    if (aposta_player_forte and aposta_banker_forte) or \
       (score['P'] >= limiar_conflito and score['B'] >= limiar_conflito and abs(score['P'] - score['B']) < diferenca_minima_para_dominancia):
        return "AGUARDAR", 50, "**Cenário de conflito**: Lógicas fortes para Player E Banker sem dominância clara. " + ("Detalhes: " + ", ".join(justificativas_finais)), "warning"
    
    limiar_aposta_tie_direta = pesos['tie_ciclo_forte'] * 1.0

    if score['T'] >= limiar_aposta_tie_direta:
        if (score['P'] < 2.0 and score['B'] < 2.0) or (abs(score['P'] - score['B']) < 1.0):
            confianca = min(100, int(score['T'] * 18))
            return "APOSTAR NO TIE", confianca, f"**ALTA CHANCE DE TIE** ({score['T']:.1f} pontos). " + ("Detalhes: " + ", ".join(justificativas_finais)), "success"
        else:
            return "AGUARDAR", 60, "**Cenário misto**: TIE forte mas Player/Banker também fortes. " + ("Detalhes: " + ", ".join(justificativas_finais)), "warning"
    
    limiar_aposta_direta_P_B = 4.0

    if aposta_player_forte and score['P'] >= limiar_aposta_direta_P_B and score['P'] > score['B'] * 1.8 and score['T'] < 1.5:
        confianca = min(100, int(score['P'] * 15))
        return "APOSTAR NO PLAYER", confianca, f"**FORTE CONSENSO para PLAYER** ({score['P']:.1f} pontos): " + ", ".join(justificativas_finais), "success"
    
    elif aposta_banker_forte and score['B'] >= limiar_aposta_direta_P_B and score['B'] > score['P'] * 1.8 and score['T'] < 1.5:
        confianca = min(100, int(score['B'] * 15))
        return "APOSTAR NO BANKER", confianca, f"**FORTE CONSENSO para BANKER** ({score['B']:.1f} pontos): " + ", ".join(justificativas_finais), "success"
    
    if justificativas_finais:
        confianca_aguardar = min(100, int(max(score.values()) * 10))
        if score['P'] > score['B'] and score['P'] > score['T']:
            detalhes_para_aguardar = f"Indicativo leve para PLAYER (score {score['P']:.1f}), mas padrões não fortes o suficiente para entrada segura."
        elif score['B'] > score['P'] and score['B'] > score['T']:
            detalhes_para_aguardar = f"Indicativo leve para BANKER (score {score['B']:.1f}), mas padrões não fortes o suficiente para entrada segura."
        elif score['T'] > score['P'] and score['T'] > score['B']:
            detalhes_para_aguardar = f"Indicativo leve para TIE (score {score['T']:.1f}), mas força insuficiente para entrada."
        else:
             detalhes_para_aguardar = "Padrões potenciais sem força ou clareza suficiente para entrada segura."

        return "AGUARDAR", confianca_aguardar, detalhes_para_aguardar + (" Detalhes: " + ", ".join(sugestoes_detalhadas)), "warning"
    
    return "AGUARDAR", 10, "Aguardando mais dados ou padrões claros. Cenário neutro.", "warning"

def exibir_analise_risco(df, confianca):
    st.subheader("📈 Análise de Risco e Confiança")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nível de Confiança", f"{confianca}%")
    with col2:
        dados_suficientes = "✅ Suficientes" if len(df) > 40 else "⚠️ Insuficientes"
        st.metric("Dados Históricos", dados_suficientes, f"{len(df)} jogos")
    with col3:
        conflitos = detectar_conflitos(df)
        nivel_conflito = "✅ Baixos" if conflitos < 2 else "⚠️ Elevados"
        st.metric("Conflitos de Padrões", nivel_conflito, f"{conflitos} conflitos")
    
    risco = max(10, 100 - confianca)
    st.caption(f"Nível de Risco: {risco}%")
    st.progress(risco/100)
    
    if risco < 30:
        st.success("**Baixo Risco**: Condições favoráveis para apostas")
    elif risco < 60:
        st.warning("**Risco Moderado**: Apostas com cautela")
    else:
        st.error("**Alto Risco**: Evite apostas ou use valores mínimos")

# --- Mostrar Análises e Sugestões ---
st.markdown("---")
st.header("📊 Análise Detalhada e Sugestões Inteligentes")

sugestoes_detalhadas_texto, logicas_ativas_map = gerar_relatorio_sugestoes(df)
acao, confianca, justificativa_acao, acao_tipo = determinar_acao_recomendada_inteligente(logicas_ativas_map, sugestoes_detalhadas_texto)

st.markdown(f"""
<div class="stAlert alert-{acao_tipo}">
    Ação Recomendada: {acao}
    <br>
    Confiança: {confianca}%
    <br>
    <span style="font-size: 0.8em; font-weight: normal;">{justificativa_acao}</span>
</div>
""", unsafe_allow_html=True)

exibir_analise_risco(df, confianca)

# Visão Geral
st.subheader("Visão Geral do Histórico")
col_overview1, col_overview2 = st.columns(2)
with col_overview1:
    st.metric("Total de Jogos Analisados", len(df))
with col_overview2:
    freq = freq_resultados(df)
    st.metric(f"Frequência Player", f"{freq['P']:.1f}%")
    st.metric(f"Frequência Banker", f"{freq['B']:.1f}%")
    st.metric(f"Frequência Tie", f"{freq['T']:.1f}%")

# Gráficos
st.subheader("Visualização dos Resultados")
if len(df) >= 1:
    freq_df = pd.DataFrame(list(freq_resultados(df).items()), columns=['Resultado', 'Porcentagem'])
    fig_freq = px.bar(freq_df, x='Resultado', y='Porcentagem', 
                      title='Frequência de Resultados (Geral)',
                      color='Resultado',
                      color_discrete_map={'P': 'blue', 'B': 'red', 'T': 'green'},
                      template="plotly_dark")
    st.plotly_chart(fig_freq, use_container_width=True)

    n_ultimos_para_grafico = min(50, len(df))
    df_ultimos = df.tail(n_ultimos_para_grafico).reset_index(drop=True)
    df_ultimos['Indice'] = df_ultimos.index + 1

    fig_seq = px.line(df_ultimos, x='Indice', y='Resultado', 
                      title=f'Sequência dos Últimos {n_ultimos_para_grafico} Resultados',
                      color='Resultado',
                      color_discrete_map={'P': 'blue', 'B': 'red', 'T': 'green'},
                      line_shape='hv',
                      markers=True,
                      template="plotly_dark")
    fig_seq.update_layout(yaxis_title="Resultado (P/B/T)", showlegend=False)
    st.plotly_chart(fig_seq, use_container_width=True)

# Sugestões Detalhadas
st.subheader("✨ Detalhes das Lógicas Ativas")
if sugestoes_detalhadas_texto:
    for s in sugestoes_detalhadas_texto:
        if "ZigZag Fraco" in s or "Alternância" in s or "conflito" in s or "cautela" in s:
            st.info(s)
        else:
            st.success(s)
else:
    st.warning("Nenhum padrão detectado no momento para detalhes.")

st.markdown("---")
st.info("Lembre-se: Este analisador é uma ferramenta de apoio. Jogos de azar envolvem risco e dependem da sorte.")

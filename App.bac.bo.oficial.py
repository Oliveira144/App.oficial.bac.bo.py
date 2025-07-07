import streamlit as st

# 🔁 Histórico com limite de 54 entradas
if "historico" not in st.session_state:
    st.session_state.historico = []

# 📌 Funções de análise
def adicionar_resultado(valor):
    st.session_state.historico.append(valor)
    if len(st.session_state.historico) > 54:
        st.session_state.historico.pop(0)

def maior_sequencia(h):
    max_seq = atual = 1
    for i in range(1, len(h)):
        if h[i] == h[i - 1]:
            atual += 1
            max_seq = max(max_seq, atual)
        else:
            atual = 1
    return max_seq

def alternancia(h):
    return sum(1 for i in range(1, len(h)) if h[i] != h[i - 1])

def eco_visual(h):
    if len(h) < 12:
        return "Poucos dados"
    return "Detectado" if h[-6:] == h[-12:-6] else "Não houve"

def dist_empates(h):
    empates = [i for i, r in enumerate(h) if r == 'E']
    return empates[-1] - empates[-2] if len(empates) >= 2 else "N/A"

def blocos_espelhados(h):
    cont = 0
    for i in range(len(h) - 5):
        if h[i:i + 3] == h[i + 3:i + 6][::-1]:
            cont += 1
    return cont

def alternancia_por_linha(h):
    linhas = [h[i:i + 9] for i in range(0, len(h), 9)]
    return [sum(1 for j in range(1, len(linha)) if linha[j] != linha[j - 1]) for linha in linhas]

def tendencia_final(h):
    ult = h[-5:]
    return f"{ult.count('C')}C / {ult.count('V')}V / {ult.count('E')}E"

def sugestao(h):
    if not h:
        return "Insira ao menos 1 resultado."
    seq = maior_sequencia(h)
    eco = eco_visual(h)
    ult = h[-1]
    if seq >= 5:
        return "🔁 Sequência longa — possível inversão"
    if ult == 'E':
        return "🟡 Empate recente — pode vir C ou V"
    if eco == "Detectado":
        return "🔄 Eco visual — padrão pode se repetir"
    return "⏳ Aguardando padrão mais claro"

# 🔵🔴🟨 Bolhas visuais
def bolha_cor(r):
    return {
        "C": "🟥",
        "V": "🟦",
        "E": "🟨"
    }.get(r, "⬜")

# 🧠 Interface principal
st.set_page_config(page_title="Football Studio – Análise", layout="wide")
st.title("🎲 Football Studio Live — Leitura de Padrões")

st.write("Adicione os resultados da rodada:")
col1, col2, col3 = st.columns(3)
if col1.button("➕ Casa (C)"):
    adicionar_resultado("C")
if col2.button("➕ Visitante (V)"):
    adicionar_resultado("V")
if col3.button("➕ Empate (E)"):
    adicionar_resultado("E")

h = st.session_state.historico

# 🧾 Histórico visual (mais recente → antigo), bolhas menores
st.subheader("🧾 Histórico visual (9 por linha, mais recente à esquerda)")
h_reverso = h[::-1]
linhas = [h_reverso[i:i + 9] for i in range(0, len(h_reverso), 9)]

for linha in linhas:
    bolhas = "".join(
        f"<span style='font-size:24px; margin-right:4px;'>{bolha_cor(r)}</span>"
        for r in linha
    )
    st.markdown(f"<div style='display:flex; gap:4px;'>{bolhas}</div>", unsafe_allow_html=True)

# 📊 Painel de análise
st.subheader("📊 Análise Preditiva")
col1, col2, col3 = st.columns(3)
col1.metric("Total Casa", h.count('C'))
col2.metric("Total Visitante", h.count('V'))
col3.metric("Total Empates", h.count('E'))

st.write(f"Maior sequência: **{maior_sequencia(h)}**")
st.write(f"Alternância total: **{alternancia(h)}**")
st.write(f"Eco visual: **{eco_visual(h)}**")
st.write(f"Distância entre últimos empates: **{dist_empates(h)}**")
st.write(f"Blocos espelhados detectados: **{blocos_espelhados(h)}**")
st.write(f"Alternância por linha: **{alternancia_por_linha(h)}**")
st.write(f"Tendência final: **{tendencia_final(h)}**")

# 🎯 Sugestão inteligente
st.subheader("🎯 Sugestão de entrada")
st.success(sugestao(h))

# 🚨 Alertas automáticos
st.subheader("🚨 Alerta estratégico")
alertas = []
if maior_sequencia(h) >= 5:
    alertas.append("🟥 Sequência longa detectada — possível inversão")
if eco_visual(h) == "Detectado":
    alertas.append("🔁 Eco visual identificado — padrão pode se repetir")
if dist_empates(h) == 1:
    alertas.append("🟠 Empates consecutivos — momento instável")
if blocos_espelhados(h) >= 1:
    alertas.append("🧩 Bloco espelhado — comportamento reflexivo")

if not alertas:
    st.info("Nenhum padrão crítico no momento.")
else:
    for alerta in alertas:
        st.warning(alerta)

# 🧹 Reset
if st.button("🧹 Limpar histórico"):
    st.session_state.historico = []
    st.rerun()

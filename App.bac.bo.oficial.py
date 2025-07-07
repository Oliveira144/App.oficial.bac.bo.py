import streamlit as st

# 🔄 Histórico com 54 posições
if "historico" not in st.session_state:
    st.session_state.historico = []

# 🧠 Funções de análise
def adicionar_resultado(valor):
    st.session_state.historico.append(valor)
    if len(st.session_state.historico) > 54:
        st.session_state.historico.pop(0)

def maior_sequencia(h):
    max_seq = atual = 1
    for i in range(1, len(h)):
        if h[i] == h[i-1]:
            atual += 1
            max_seq = max(max_seq, atual)
        else:
            atual = 1
    return max_seq

def alternancia(h):
    return sum(1 for i in range(1, len(h)) if h[i] != h[i-1])

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
        if h[i:i+3] == h[i+3:i+6][::-1]:
            cont += 1
    return cont

def alternancia_por_linha(h):
    linhas = [h[i:i+18] for i in range(0, len(h), 18)]
    return [sum(1 for j in range(1, len(linha)) if linha[j] != linha[j-1]) for linha in linhas]

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
        return "🟡 Empate recente — próxima rodada pode ser C ou V"
    if eco == "Detectado":
        return "🔄 Eco visual — repetir padrão anterior"
    return "⏳ Aguardar novo padrão"

# 🌐 Interface Streamlit
st.set_page_config(page_title="Football Studio Análise", layout="wide")
st.title("🎲 Football Studio Live - Análise Inteligente")

st.write("Insira os resultados conforme forem aparecendo no jogo:")
col1, col2, col3 = st.columns(3)
if col1.button("➕ Casa (C)"): adicionar_resultado("C")
if col2.button("➕ Visitante (V)"): adicionar_resultado("V")
if col3.button("➕ Empate (E)"): adicionar_resultado("E")

st.markdown("---")
h = st.session_state.historico

# 🧾 Histórico visual estilo matriz
st.subheader("🧾 Histórico (54 posições)")
for i in range(0, len(h), 18):
    linha = " | ".join(h[i:i+18])
    st.code(linha)

# 🔍 Análises
st.subheader("📈 Análise Preditiva")
col1, col2, col3 = st.columns(3)
col1.metric("Total C", h.count('C'))
col2.metric("Total V", h.count('V'))
col3.metric("Total E", h.count('E'))

st.write(f"Maior sequência: {maior_sequencia(h)}")
st.write(f"Alternância total: {alternancia(h)}")
st.write(f"Eco visual: {eco_visual(h)}")
st.write(f"Distância entre últimos empates: {dist_empates(h)}")
st.write(f"Blocos espelhados detectados: {blocos_espelhados(h)}")
st.write(f"Alternância por linha: {alternancia_por_linha(h)}")
st.write(f"Tendência final: {tendencia_final(h)}")

# 🔮 Sugestão de jogada
st.subheader("🎯 Sugestão de entrada")
st.success(sugestao(h))

# 🧹 Reset (opcional)
if st.button("🧹 Limpar histórico"):
    st.session_state.historico = []
    st.rerun()

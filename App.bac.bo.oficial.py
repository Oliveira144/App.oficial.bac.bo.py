import streamlit as st
from collections import Counter

# Inicializa histórico
if "historico" not in st.session_state:
    st.session_state.historico = []

# Funções de lógica
def cores_opostas(c1, c2):
    return (c1 == "🔴" and c2 == "🔵") or (c1 == "🔵" and c2 == "🔴")

def padrao_reescrito(linha1, linha2):
    if len(linha1) != len(linha2):
        return False
    for a, b in zip(linha1, linha2):
        if a == "🟡" or b == "🟡":
            continue
        if not cores_opostas(a, b):
            return False
    return True

def colunas_semelhantes(c1, c2):
    for a, b in zip(c1, c2):
        if a == "🟡" or b == "🟡":
            continue
        if not cores_opostas(a, b):
            return False
    return True

def inserir(cor):
    st.session_state.historico.insert(0, cor)

def desfazer():
    if st.session_state.historico:
        st.session_state.historico.pop(0)

def limpar():
    st.session_state.historico.clear()

# Configuração visual
st.set_page_config(page_title="FS Análise Pro", layout="centered")

st.title("📊 FS Análise Pro")
st.caption("Detecção de padrões reescritos e sugestões inteligentes para o jogo Football Studio Live")

# Botões de entrada
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔴 Casa", use_container_width=True): inserir("🔴")
with col2:
    if st.button("🔵 Visitante", use_container_width=True): inserir("🔵")
with col3:
    if st.button("🟡 Empate", use_container_width=True): inserir("🟡")

# Controles
col4, col5 = st.columns(2)
with col4:
    if st.button("↩️ Desfazer", use_container_width=True): desfazer()
with col5:
    if st.button("🧹 Limpar", use_container_width=True): limpar()

# Exibir histórico
st.divider()
st.subheader("📋 Histórico de Jogadas (blocos de 9, direita → esquerda, mais recentes no topo)")

linhas = []
for i in range(0, len(st.session_state.historico), 9):
    linha = st.session_state.historico[i:i+9]
    linha = linha[::-1]  # direita → esquerda
    linhas.append(linha)

linhas_exibidas = linhas[::-1]  # mais recentes no topo
for idx, linha in enumerate(linhas_exibidas):
    st.markdown(f"**Linha {len(linhas_exibidas)-idx}:** " + " ".join(linha))

# Frequência
st.divider()
st.subheader("📊 Frequência de Cores")
contagem = Counter(st.session_state.historico)
st.write(f"🔴 Casa: {contagem['🔴']} | 🔵 Visitante: {contagem['🔵']} | 🟡 Empate: {contagem['🟡']}")

# Análise de padrão reescrito
st.divider()
st.subheader("🧠 Detecção de Padrão Reescrito")

linhas_validas = [l for l in linhas_exibidas if len(l) == 9]

if len(linhas_validas) >= 2:
    linha1 = linhas_validas[0]  # Mais recente
    linha2 = linhas_validas[1]  # Segunda mais recente

    if padrao_reescrito(linha1, linha2):
        ultima_jogada = linha1[-1]
        jogada_sugerida = "🔵" if ultima_jogada == "🔴" else "🔴" if ultima_jogada == "🔵" else "❓"
        st.success(f"""
        🔁 **Padrão reescrito com inversão cromática detectado!**
        \nÚltima jogada: {ultima_jogada}
        \n🎯 **Sugestão:** Jogar {jogada_sugerida} (oposto à última)
        """)
    else:
        st.info("⏳ Nenhum padrão reescrito identificado entre as duas últimas linhas completas.")
elif len(st.session_state.historico) < 18:
    st.warning("⚠️ Registre pelo menos 18 jogadas para ativar a análise (2 linhas de 9).")
else:
    st.info("Aguardando segunda linha completa para análise.")

# Análise por colunas verticais
st.divider()
st.subheader("🧬 Análise por Colunas Verticais")

ultimos_27 = st.session_state.historico[:27]

if len(ultimos_27) == 27:
    linhas_3x9 = []
    for i in range(0, 27, 9):
        linha = ultimos_27[i:i+9][::-1]
        linhas_3x9.append(linha)

    colunas = list(zip(*linhas_3x9))  # 9 colunas de 3 cores cada

    ref_coluna_antiga = colunas[3]
    nova_coluna = colunas[0]

    if colunas_semelhantes(ref_coluna_antiga, nova_coluna):
        coluna_apos_ref = colunas[4]
        proxima_sugestao = coluna_apos_ref[0]
        sugestao_convertida = "🔵" if proxima_sugestao == "🔴" else "🔴" if proxima_sugestao == "🔵" else "❓"

        st.success(f"""
        🔂 Estrutura de colunas repetida com troca de cores detectada!
        \n📌 Coluna antiga (posição 4) ≈ Nova coluna (posição 1)
        \n🎯 Sugestão baseada na coluna que seguiu a referência anterior:
        **{sugestao_convertida}**
        """)
    else:
        st.info("📊 Nenhum padrão repetido de colunas encontrado nas últimas 27 jogadas.")
else:
    st.warning("⚠️ Registre pelo menos 27 jogadas para ativar a análise por colunas verticais.")

# Visualização das colunas
if len(ultimos_27) == 27:
    st.subheader("🧱 Visualização das Colunas Verticais (3x9)")

    col_container = st.container()
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = col_container.columns(9)
    colunas_texto = list(zip(*linhas_3x9))

    for i, coluna in enumerate(colunas_texto):
        texto = f"**Coluna {i+1}**\n" + "\n".join(coluna)
        match i:
            case 0: col1.markdown(texto)
            case 1: col2.markdown(texto)
            case 2: col3.markdown(texto)
            case 3: col4.markdown(texto)
            case 4: col5.markdown(texto)
            case 5: col6.markdown(texto)
            case 6: col7.markdown(texto)
            case 7: col8.markdown(texto)
            case 8: col9.markdown(texto)

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configurar a página inicial
st.set_page_config(layout="wide", page_title="Airbnb Rio de Janeiro - Previsão de Preços")

# Função para carregar o modelo
@st.cache_resource
def carregar_modelo():
    return joblib.load('../Results/modelo_random_forest.pkl')

# Seleção da página no menu lateral
menu = st.sidebar.radio(
    "Menu",
    options=["Previsão de Preços", "Análise Exploratória (Notebook)", "Sobre o Projeto"]
)

# Página de Previsão de Preços
if menu == "Previsão de Preços":
    st.title("Estimativa de Preço de Diária para Imóveis no Rio de Janeiro")
    st.write("Insira as características do seu imóvel e obtenha uma estimativa de preço da diária.")

    modelo = carregar_modelo()

    # Função para preparar os dados do usuário
    def preparar_dados_usuario(bairro, tipo_acomodacao, acomodantes, quartos, banheiros, camas, wifi, estacionamento, ar_condicionado, cozinha, tipo_acomodacao_usuario):
        colunas_treinamento = modelo.feature_names_in_
        dados_usuario_dict = {
            'accommodates': acomodantes,
            'bedrooms': quartos,
            'bathrooms': banheiros,
            'beds': camas,
            'has_wifi': int(wifi),
            'has_estacionamento': int(estacionamento),
            'has_ar_condicionado': int(ar_condicionado),
            'has_cozinha': int(cozinha),
            'tipo_acomodacao_usuario': tipo_acomodacao_usuario
        }

        for coluna in colunas_treinamento:
            if coluna.startswith("neighbourhood_cleansed_"):
                dados_usuario_dict[coluna] = 1 if coluna == f"neighbourhood_cleansed_{bairro.lower()}" else 0
            elif coluna.startswith("tipo_acomodacao_"):
                dados_usuario_dict[coluna] = 1 if coluna == f"tipo_acomodacao_{tipo_acomodacao.lower()}" else 0

        dados_usuario = pd.DataFrame([dados_usuario_dict])
        dados_usuario = dados_usuario.reindex(columns=colunas_treinamento, fill_value=np.nan)
        return dados_usuario

    # Interface do usuário
    st.write("")  # Espaçamento
    col1, col2 = st.columns([1.2, 2])  # Proporções ajustadas para mais espaço no mapa
    with col1:
        st.write("**Características do Imóvel**")
        bairro = st.selectbox("Bairro", ["Copacabana", "Ipanema", "Barra da Tijuca", "Leblon", "Jacarepaguá", "Outros"])
        tipo_acomodacao = st.selectbox("Tipo de Acomodação", ["Casa", "Apartamento", "Loft", "Studio"])
        tipo_acomodacao_usuario = st.selectbox("Classificação do Imóvel", ["Econômico", "Confortável", "Luxo"])
        acomodantes = st.slider("Acomodantes", min_value=1, max_value=10, value=2)
        quartos = st.slider("Quartos", min_value=1, max_value=5, value=1)
        banheiros = st.slider("Banheiros", min_value=1, max_value=5, value=1)
        camas = st.slider("Camas", min_value=1, max_value=10, value=1)
        wifi = st.checkbox("Wi-Fi")
        estacionamento = st.checkbox("Estacionamento Gratuito")
        ar_condicionado = st.checkbox("Ar-Condicionado")
        cozinha = st.checkbox("Cozinha")

        dados_usuario = preparar_dados_usuario(
            bairro, tipo_acomodacao, acomodantes, quartos, banheiros, camas, wifi, estacionamento, ar_condicionado, cozinha, tipo_acomodacao_usuario
        )

        if st.button("Prever Preço"):
            preco_estimado = modelo.predict(dados_usuario.fillna(0))
            st.success(f"O preço estimado da diária é **R$ {preco_estimado[0]:.2f}**")


    with col2:
        st.subheader("Mapa de Calor")
        st.write("Este mapa de densidade apresenta a distribuição dos bairros do Rio de Janeiro, destacando os maiores e menores valores de diárias de hospedagem.")
        st.write("- **Baixa densidade (azul):** Preços baixos")
        st.write("- **Alta densidade (vermelho):** Preços altos")
        with open("../Results/mapa_precos_airbnb.html", "r", encoding="utf-8") as file:
            mapa_html = file.read()
        st.components.v1.html(mapa_html, width=800, height=500)

# Página de Análise Exploratória
elif menu == "Análise Exploratória (Notebook)":
    st.title("Análise Exploratória de Dados")
    st.write("Abaixo está o conteúdo do Jupyter Notebook utilizado no projeto:")
    with open("../Results/airbnb_exploratory_analysis.html", "r", encoding="utf-8") as file:
        notebook_html = file.read()
    st.components.v1.html(notebook_html, width=1500, height=34200)


# Página Sobre o Projeto
elif menu == "Sobre o Projeto":
    st.title("Sobre o Projeto")
    st.write("""
    **Este projeto foi desenvolvido com finalidade exclusivamente acadêmica e para aprimoramento de estudos na área de Ciência de Dados.**  
    Não possui qualquer objetivo comercial ou vínculo oficial com o Airbnb ou outras empresas.  

    O principal intuito é aplicar técnicas de análise exploratória, engenharia de features e modelagem preditiva para consolidar conceitos e práticas da área.  
    A análise foi construída utilizando dados públicos do Airbnb (de abril de 2018 a maio de 2020) de imóveis localizados no Rio de Janeiro.  

    Neste projeto, foi possível praticar a utilização de modelos supervisionados de Machine Learning e explorar diversas ferramentas para criar uma solução completa.
    """)

    st.subheader("Ferramentas Utilizadas")
    st.write("- **Python** para manipulação e análise de dados")
    st.write("- **Streamlit** para construção da aplicação")
    st.write("- **Scikit-learn** para modelagem preditiva")
    st.write("- **Folium** para mapas interativos")
    st.write("- **Git** para controle de versão")
    st.write("- **GitHub** como repositório e colaboração")

    st.subheader("Contato")
    st.write("Para mais informações, veja o repositório completo no GitHub: [GitHub/Jonathan Mesquita](https://github.com/JONATHAN-ALMEIDA-MESQUITA/Predictive_analysis_Airbnb).")
    st.write("Caso queira se conectar, visite meu perfil no LinkedIn: [LinkedIn/Jonathan Mesquita](https://www.linkedin.com/in/jonathan-mesquita-3049581b1/).")

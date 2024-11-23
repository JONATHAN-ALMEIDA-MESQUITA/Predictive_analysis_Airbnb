# Use uma imagem base do Python
FROM python:3.9-slim

# Configure o diretório de trabalho
WORKDIR /app

# Copie os arquivos da pasta streamlit_deploy para o container
COPY . /app/

# Instale as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta usada pelo Streamlit
EXPOSE 8501

# Comando para rodar o Streamlit com o script correto
CMD ["streamlit", "run", "airbnb_predictive_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

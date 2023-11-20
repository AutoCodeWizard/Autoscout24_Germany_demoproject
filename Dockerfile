# Verwenden Sie ein offizielles Python-Runtime-Image als Basis
FROM python:3.11-slim

# Setzen Sie das Arbeitsverzeichnis im Container
WORKDIR /app

# Installieren Sie die benötigten Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopieren Sie die Projektdateien in den Container
COPY . /app

# Führen Sie den Streamlit-Server aus
CMD ["streamlit", "run", "app.py", "--server.runOnSave=true"]

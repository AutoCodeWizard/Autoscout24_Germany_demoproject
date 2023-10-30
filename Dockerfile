# Verwenden Sie ein offizielles Python-Laufzeitimage als Basisimage
FROM python:3.11.5

# Setzen des Arbeitsverzeichnisses im Container auf /app
WORKDIR /app

# Kopieren der aktuellen Verzeichnisinhalte in den Container unter /app
COPY . /app

# Installieren der Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit-App starten
CMD ["streamlit", "run", "app.py"]

# Autoscout24_Germany_demoproject

![Python](https://img.shields.io/badge/Python-3.11.5-blue)
![Framework](https://img.shields.io/badge/Streamlit-1.27-yellow)
![License](https://img.shields.io/badge/MIT-License-green)

## Inhalt

- [Voraussetzungen](#voraussetzungen)
  - [Empfohlene Entwicklungsumgebung](#empfohlene-entwicklungsumgebung)
- [Über das Projekt](#über-das-projekt)
  - [Verwendete Technologien](#verwendete-technologien)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Beitrag](#beitrag)
- [Lizenz](#lizenz)
- [Kontakt](#kontakt)

## Voraussetzungen

Bevor Sie mit dem Projekt beginnen, stellen Sie sicher, dass die folgenden Tools auf Ihrem System installiert sind:
- Python 3.11.5 oder höher
- Git

### Empfohlene Entwicklungsumgebung

- [Visual Studio Code](https://code.visualstudio.com/)
  - Erweiterungen:
    - Python
    - Jupyter

Um die Erweiterungen zu installieren, können Sie den Extension Marketplace in Visual Studio Code nutzen oder die folgenden Befehle im Terminal ausführen:
```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

## Über das Autoscout24_Germany_demoproject

Dieses Projekt dient der Analyse und Visualisierung von Daten von Autoscout24.de. Es wird hauptsächlich für Datenwissenschaftliche Zwecke verwendet.

### Verwendete Technologien

- Python 3.11.5
- Streamlit
- Jupyter Notebook

## Installation

### Klonen Sie das Repository:
```bash
git clone https://github.com/AutoCodeWizard/Autoscout24_Germany_demoproject.git
```

### Wechseln Sie in das Projektverzeichnis:
```bash
cd Autoscout24_Germany_demoproject
```
### weiter im Container mit Docker (optional, siehe unten) oder:
### Erstellen Sie eine virtuelle Umgebung: 
```bash
python -m venv venv
```

### Aktivieren Sie die virtuelle Umgebung:
```bash
source venv/bin/activate  # Für Linux und macOS
```
```bash
venv\\Scripts\\Activate  # Für Windows
```

### Installieren Sie die Abhängigkeiten in der neuen Python virtuellen Umgebung:
```bash
pip install -r requirements.txt
```

### oder falls Sie conda verwenden wollen:
```bash 
conda create --name myenv python=3.11.5
```
```bash
conda activate myenv
```
```bash
conda install --file requirements.txt
```


## Verwendung

### Web-App:
```bash
streamlit run app.py
```
### im Container mit Docker (optional)

Wenn Sie Docker installiert haben, können Sie die Anwendung auch in einem Docker-Container ausführen.

1. Bauen Sie das Docker-Image:

    ```bash
    docker build -t autoscout24_germany_demo .
    ```

2. Starten Sie den Docker-Container:

    ```bash
    docker run -p 8501:8501 autoscout24_germany_demo
    ```

    Die Streamlit-App sollte jetzt unter `http://localhost:8501` erreichbar sein.

## Beiträge

Beiträge sind willkommen! Für größere Änderungen öffnen Sie bitte zuerst ein Issue, um zu diskutieren, was Sie ändern möchten.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Weitere Informationen finden Sie in der Datei `LICENSE.txt`

## Kontakt

Lukas Hamann - https://www.linkedin.com/in/lukas-hamann-76b736295

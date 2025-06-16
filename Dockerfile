# Étape 1 : image de base officielle Python
FROM python:3.10-slim

# Étape 2 : Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Étape 3 : définir le répertoire de travail
WORKDIR /app

# Étape 4 : copier les fichiers dans l'image Docker
COPY requirements.txt .

# Étape 5 : installer les packages Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 6 : copier le reste du code
COPY . .

# Étape 7 : exposer le port Streamlit (par défaut 8501)
EXPOSE 8501

# Étape 8 : lancer Streamlit
CMD ["streamlit", "run", "src/streamlit/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
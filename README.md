# 🚀 Universal ML Pipeline - Production-Ready System

Un sistema completo di Machine Learning end-to-end, dalla preparazione dati al deployment API production-ready.

## 🎯 Overview

Questo progetto implementa un **pipeline ML universale** che gestisce automaticamente:
- 📊 **Data preprocessing** con gestione valori mancanti e encoding
- 🤖 **Model training** con grid search e cross-validation
- 🚀 **API deployment** con server Flask production-ready
- 🧪 **Testing completo** con validation automatica

### 🏆 Risultati Chiave
- **99.5% Accuracy** su dataset multiclass
- **<50ms Response Time** per predizioni API
- **100% Test Success Rate** su validation automatica
- **Zero Data Leakage** con architettura corretta

## 🚀 Quick Start

### 1️⃣ Setup Ambiente
```bash
# Clone repository
git clone https://github.com/seb0t/universal_project_for-data_prediction.git
cd universal_project_for-data_prediction

# Crea virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Installa dipendenze
pip install -r requirements.txt
```

### 2️⃣ Esegui Data Processing
```bash
# Apri Jupyter
jupyter notebook

# Esegui: notebooks/01_data_exploration_clean.ipynb
# → Preprocessing automatico e split dati
```

### 3️⃣ Training Modello
```bash
# Esegui: notebooks/02_model_training.ipynb  
# → Grid search, training, evaluation automatica
# → Salva modello finale + metadata
```

### 4️⃣ Deploy API Server
```bash
# Avvia server API
python api_server.py

# Server attivo su: http://127.0.0.1:5000
```

### 5️⃣ Test API
```bash
# Esegui: notebooks/03_api_test.ipynb
# → Test automatico tutti gli endpoints
# → Validation con dati reali

# Oppure usa Postman/curl:
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "gender": "Female", "sleep_quality_index": 6.5}'
```

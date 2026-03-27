# 🏥 voize Healthcare Pipeline — Azure Deployment

## Architecture

```
Google Colab          GitHub + DVC           Azure
────────────          ────────────           ─────

Train NER
    ↓
Save pkl
    ↓
dvc push ──────────→  Google Drive (pkl)
git push ──────────→  GitHub Actions triggers
                            ↓
                       Pull pkl from Drive
                            ↓
                       Quality gate (F1 >= 0.75?)
                            ↓
                       Run tests
                            ↓
                       docker build
                       (FastAPI + Streamlit + pkl)
                            ↓
                       Push to ACR ──────────────→  Azure Container Registry
                            ↓
                       Deploy to ACI ────────────→  Azure Container Instances
                                                     http://YOUR_IP:8000  FastAPI
                                                     http://YOUR_IP:8501  Streamlit
```

---

## Azure resources used

| Resource | What it does | Cost |
|---|---|---|
| **Resource Group** | Groups all resources together | Free |
| **Azure Container Registry (ACR)** | Stores your Docker images | ~$5/month (Basic) |
| **Azure Container Instances (ACI)** | Runs your container | ~$0.0025/vCPU/hour |

---

## Complete setup — step by step

### Step 1 — Install Azure CLI
```bash
# Windows (in Anaconda Prompt):
winget install Microsoft.AzureCLI

# Verify
az --version
```

### Step 2 — Login to Azure
```bash
az login
# Browser opens → sign in with your Azure account
```

### Step 3 — Run Azure setup script (creates all resources)
```bash
chmod +x scripts/azure_setup.sh
./scripts/azure_setup.sh

# This creates:
#   Resource Group: voize-rg
#   Container Registry: voizehealthcare.azurecr.io
#   Service Principal for GitHub Actions
```

### Step 4 — Add GitHub Secrets
```
Go to: GitHub repo → Settings → Secrets → Actions

Add these secrets:

AZURE_CREDENTIALS          → JSON from azure_setup.sh output
GDRIVE_CREDENTIALS_DATA    → Google Drive service account JSON
```

### Step 5 — First time Git + DVC setup
```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/voize-healthcare.git
dvc init
dvc remote modify gdrive url gdrive://YOUR_GDRIVE_FOLDER_ID
git add .
git commit -m "initial setup"
git push -u origin main
```

### Step 6 — Copy pkl from Colab and deploy
```bash
# Copy ner_model.pkl → models/ner_model.pkl
# Copy metrics.json  → evaluation/metrics.json

make dvc-push
# This:
# 1. Tracks pkl with DVC
# 2. Uploads pkl to Google Drive
# 3. Pushes to GitHub → CI/CD triggers
# 4. Azure gets new Docker image automatically
```

---

## Every time you train a new model

```bash
# 1. Train on Colab
# 2. Download ner_model.pkl and metrics.json
# 3. Copy into project:
#    models/ner_model.pkl
#    evaluation/metrics.json
# 4. Run:
make dvc-push
# Everything else is automatic
```

---

## Check your deployed app

```bash
# Get the Azure IP address
make azure-url

# Check container status
make azure-status

# View live logs
make azure-logs
```

---

## Run locally first (to test before deploying)

```bash
# Option 1: Without Docker
pip install -r requirements.txt
make run-api   # Terminal 1
make run-ui    # Terminal 2

# Option 2: With Docker
make docker-build
make docker-run
```

---

## Project structure

```
voize-azure/
├── app/
│   ├── pipeline.py        ← loads pkl + Whisper
│   ├── main.py            ← FastAPI
│   └── streamlit_app.py   ← Streamlit UI
│
├── models/
│   └── ner_model.pkl.dvc  ← DVC pointer (triggers CI/CD when changed)
│
├── evaluation/
│   └── metrics.json       ← real F1 metrics from Colab
│
├── scripts/
│   ├── azure_setup.sh     ← run once to create Azure resources
│   └── check_quality.py   ← quality gate
│
├── tests/
│   └── test_api.py
│
├── .dvc/config            ← Google Drive remote
├── .github/workflows/
│   └── cicd.yml           ← CI/CD: DVC → quality gate → Docker → ACR → ACI
│
├── Dockerfile             ← ONE container: FastAPI + Streamlit
├── supervisord.conf       ← starts both services
├── Makefile               ← all commands
└── requirements.txt
```

.PHONY: install run-api run-ui docker-build docker-run azure-setup azure-logs azure-url dvc-push test

# ── Local setup ────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Local dev (no Docker) ──────────────────────────────────────────────────
run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	API_URL=http://localhost:8000 streamlit run app/streamlit_app.py --server.port 8501

# ── git-Setup ──────────────────────────────────────────────────
git init
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/voize-healthcare.git


# ── DVC -Setup ──────────────────────────────────────────────────
dvc init

dvc remote add -d gdrive gdrive://YOUR_GDRIVE_FOLDER_ID


git add .
git commit -m "initial setup"
git push -u origin main

# ── DVC push and git push──────────────────────────────────────────────────
dvc add models/ner_model.pkl
dvc push --remote gdrive
git add models/ner_model.pkl.dvc
git commit -m "new model"
git push

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v



# ── Step 6: Build Docker image ──────────────────────────────────────────
      - name: Build image
        run: docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest .

 # ── Step 6: Login to Azure  ──────────────────────────────────────────

    docker login <acr-name>.azurecr.io

# ── Step 7: Push to Azure Container Registry ────────────────────────────
    - name: Push image
      run: docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
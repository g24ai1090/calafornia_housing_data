# calafornia_housing_data
calafornia_housing_data_assignment3

This repository implements an **end-to-end MLOps pipeline** for a regression model trained on the **California Housing dataset**. The project is split across four branches, each representing a different stage of the workflow.

##  Branch Structure

- **`main`** → Base branch with initial setup (`README.md`, `.gitignore`).
- **`dev`** → Model development (training a scikit-learn Linear Regression model).
- **`docker_ci`** → Docker containerization and CI/CD setup using GitHub Actions.
- **`quantization`** → Manual quantization and PyTorch model creation.

## ⚙️ 1. `dev` Branch – Model Development

- Trains a **Linear Regression** model on the **California Housing** dataset using `scikit-learn`.
- Saves the model as `model.joblib`.

git checkout -b dev
conda create -n calafornia_data python=3.10
conda activate calafornia_data
pip install -r requirements.txt
python train.py
This will train the model and save:

model.joblib – full sklearn model.

## 2. docker_ci Branch – Containerization & CI/CD
Adds a Dockerfile to containerize the model.

Adds predict.py to load the model and make a sample prediction.

Adds .github/workflows/ci.yml to:

-Train the model.
-Build the Docker image.
-Run the container to verify predict.py.
-Push the image to DockerHub.

git checkout -b docker_ci (on top of dev)
git add --all
git commit -m ""
git push origin docker_ci 
This will trigger workflow and push docker image to repo

## 3. quantization Branch – Model Conversion & Optimization
Loads the trained scikit-learn model.
Extracts coef_ and intercept_.
Performs manual quantization (FP32 → FP16).
Creates a single-layer PyTorch model and sets its weights from sklearn’s parameters.

Runs inference using dequantized weights.

# How to run:

git checkout -b quantization
pip install -r requirements.txt (to install torch)
python quantize.py
# This will generate:
unquant_params.joblib – Original model parameters.
quant_params.joblib – Quantized model parameters.
quantized_model.pth – PyTorch model with quantized weights.

# Quantization Results
--------------------------------------------
Metric	| Original Model	| Quantized Model
-------------------------------------------
R² Score |	0.6053	         |  0.6051
---------------------------------------------
Model Size  |	0.40 KB	      |  0.32 KB
-----------------------------------------------
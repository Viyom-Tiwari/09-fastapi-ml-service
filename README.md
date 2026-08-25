# FastAPI ML Service

An end-to-end ML application exposing a trained tabular classifier through FastAPI with typed request validation, health checks, and a small test suite.

## Why this project

This repository is an original, reproducible portfolio project designed to demonstrate practical engineering rather than notebook-only experimentation.

## Stack

Python, scikit-learn, FastAPI, Uvicorn

## Architecture

```mermaid
flowchart LR
    A[Input data] --> B[Validation and preprocessing]
    B --> C[Model or retrieval layer]
    C --> D[Evaluation]
    D --> E[Prediction or API output]
```

The service trains a small model at import time for zero-setup local execution; production deployments should load a versioned artifact during startup instead. Pydantic validates input shape before inference.

## Live demo

The browser demo is deployed on Vercel at [09-fastapi-ml-service-demo](https://viyom-public-ml-demo-m2kpvs94d-viyom1.vercel.app). The deployed page links back to the [GitHub source repository](https://github.com/Viyom-Tiwari/09-fastapi-ml-service).

## Data and APIs

Uses scikit-learn’s built-in iris dataset, so the service runs offline and requires no credentials.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
uvicorn src.api:app --reload
```

## Reproducibility notes

The implementation uses fixed random seeds where randomness is involved, keeps training and inference paths separate, and reports metrics rather than making unsupported performance claims.

## Project structure

```text
09-fastapi-ml-service/
├── api/                 # Vercel-compatible FastAPI entrypoint
├── src/                 # local implementation
├── tests/               # lightweight verification
├── README.md
├── requirements.txt
└── .gitignore
```

## Next improvements

Add experiment tracking, richer data validation, and deployment monitoring as the project evolves.

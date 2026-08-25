# FastAPI ML Service

A runnable HTTP service around an iris classifier, with typed request validation, a health endpoint, a browser demo, and a Vercel-compatible `api/index.py` entrypoint.

## Live demo

The browser demo is available at [Vercel deployment](https://viyom-public-ml-demo-m2kpvs94d-viyom1.vercel.app). Its UI links back to this [GitHub repository](https://github.com/Viyom-Tiwari/09-fastapi-ml-service).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api:app --reload
pytest -q
```

Then open `http://127.0.0.1:8000/docs` for the generated API documentation. A sample request is:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Architecture

`src/api.py` serves the local app, while `api/index.py` is the serverless entrypoint used by Vercel. Both expose the same typed feature contract and reject non-positive measurements before inference.

## Data and limitations

The model uses scikit-learn’s built-in iris dataset and is an educational example, not a botanical or scientific decision service. Production use would load a versioned artifact during startup, validate a formal feature schema, and add authentication and monitoring.

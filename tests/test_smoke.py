from fastapi.testclient import TestClient
from src.api import app

def test_health(): assert TestClient(app).get("/health").status_code==200

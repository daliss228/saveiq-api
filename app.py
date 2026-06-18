from fastapi import FastAPI
from services.prediction_service import router

app = FastAPI()

app.include_router(router)

# Health Check

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SaveIQ API funcionando"
    }

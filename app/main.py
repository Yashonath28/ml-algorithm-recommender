from fastapi import FastAPI
from app.routes.analyze import router as analyze_router

app = FastAPI(title="ML Algorithm Recommender API")

app.include_router(analyze_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Welcome to the ML Algorithm Recommender API!"}
from fastapi import FastAPI
from .models import UserModel
from .database import engine, SessionLocal

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
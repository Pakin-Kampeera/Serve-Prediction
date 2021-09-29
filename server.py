from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from script.word_embedding_model import predict

app = FastAPI(redoc_url=None)


class Series(BaseModel):
    series: list


@app.post('/prediction', tags=['Prediction'])
async def stress_prediction(series: Series):
    return await predict(series)

if __name__ == '__main__':
    uvicorn.run('server:app', host='0.0.0.0', reload=True, port=8000)

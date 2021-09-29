FROM tiangolo/uvicorn-gunicorn:python3.8

WORKDIR /app

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "python", "server.py"]
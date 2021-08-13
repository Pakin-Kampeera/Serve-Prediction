FROM tiangolo/uvicorn-gunicorn:python3.8

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn

EXPOSE 80

CMD [ "uvicorn", "server:app", "--port", "80"]
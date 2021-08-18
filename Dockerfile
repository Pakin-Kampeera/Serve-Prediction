FROM tiangolo/uvicorn-gunicorn:python3.8

WORKDIR /app

COPY . .

RUN python3 -m venv serve_pred

RUN source serve_pred/bin/activate

RUN pip install -r requirements.txt

EXPOSE 80

CMD [ "python", "server.py"]
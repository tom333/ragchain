FROM python:3.11-slim-buster

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
COPY app.py .

RUN pip install -r requirements.txt

CMD ["chainlit", "run", "app.py"]
FROM python:3.11-slim-buster

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
COPY app.py .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["chainlit", "run", "app.py"]
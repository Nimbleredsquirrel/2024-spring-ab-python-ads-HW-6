FROM python:3.11.8-slim-bullseye

EXPOSE 8501

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY app.py app.py

CMD [ "streamlit", "run", "/app.py", "--server.port", "8501"]

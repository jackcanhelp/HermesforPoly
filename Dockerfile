FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir ddgs beautifulsoup4 streamlit plotly

COPY . .

RUN mkdir -p data

ENV DATA_DIR=data

CMD ["python", "main.py"]

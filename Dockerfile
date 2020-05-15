FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN pip install --no-cache-dir spacy gensim

RUN python -m spacy download en_core_web_sm

# Gunicorn configuration
ENV MAX_WORKERS 2
ENV WEB_CONCURRENCY 2

COPY ./app /app

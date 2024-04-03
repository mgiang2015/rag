FROM python:3.11-slim

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

# Copy dependencies

COPY ./.env ./.env

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

COPY ./requirements.txt /code/requirements.txt

# Install dependencies

RUN poetry install  --no-interaction --no-ansi --no-root

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080

# docker build -t rag-pinecone .
# docker tag rag-pinecone mgiang2015/rag-pinecone
# docker push mgiang2015/rag-pinecone
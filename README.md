# Retrieval-Augmented Generation (RAG) Project

With more widespread use of Large Language Models (LLMs), the risks of hallucination becomes significant. One approach to minimise hallucination is using [Retrieval-Augmented Generation](https://aws.amazon.com/what-is/retrieval-augmented-generation/) technique, where the LLM knowledge is bounded in a given context. Context comes from ingesting material (websites, documents, text, ...), creating embeddings and storing them in a vector database.

## Demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=liBhf8EDQKs" target="_blank"><img src="http://img.youtube.com/vi/liBhf8EDQKs/0.jpg" alt="Demo video on Youtube" width="480" height="360" border="10" /></a>

## Tech Stack
Backend: [Python](https://www.python.org/)

Frameworks: [LangChain](https://www.langchain.com/), [FastAPI](https://fastapi.tiangolo.com/)

Vectorstore: [Pinecone](https://www.pinecone.io/)

## Environment Set-up

1. Clone the repository and navigate to my-app directory

```
cd my-app/
```

2. Set up your [python virtual environment](https://docs.python.org/3/library/venv.html) and install dependencies

```
pip install -r requirements.txt
```

3. Set up your [Pinecone account](https://docs.pinecone.io/guides/getting-started/quickstart), follow their documentation to set up [a project](https://docs.pinecone.io/guides/projects/create-a-project) and [an index](https://docs.pinecone.io/guides/indexes/create-an-index)

4. Set up your [OpenAI API account](https://platform.openai.com/docs/quickstart?context=python)

5. (Optional) Set up your [LangSmith account](https://docs.smith.langchain.com/setup) if you'd like to trace LangChain's output at every step.

6. Set up your environment variables

```
# Compulsory
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
PINECONE_INDEX=
OPENAI_API_KEY=

# Optional
LANGCHAIN_TRACING_V2=
LANGCHAIN_ENDPOINT=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=
```

7. Run the application at port 8080 (change port if needed)

```
uvicorn app.server:app --host 0.0.0.0 --port 8080
```

8. Navigate to `/docs` for documentation of API endpoints offered.

9. Main end-points you will interact with are:

- `POST /rag-pinecone/website`: Ingest data from a given website. Takes a query parameter `url`. Example:

```
http://localhost:8080/rag-pinecone/website?url=https://lilianweng.github.io/posts/2023-06-23-agent
```

- `POST /rag-pinecone/invoke`: Question and answer with agent. Takes in a JSON body.
```
{
    "input": "What is task decomposition?"
}
```

- `POST /rag-pinecone/stream`: Question and answer with agent, but answers are streamed. Takes in a JSON body
```
{
    "input": "What is task decomposition?"
}
```

10. Feel free to modify `packages/rag-pinecone/rag_pinecone/chain.py` to experiment with LangChain. With every modification, run the following to update the module:
```
pip install packages/rag-pinecone
```

## Contacts

Email: mgiang2015@gmail.com

LinkedIn: https://www.linkedin.com/in/leminhgiang/

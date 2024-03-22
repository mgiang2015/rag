from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_pinecone import chain as rag_pinecone_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

import io
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Blob
from langchain.document_loaders.parsers import PDFMinerParser

from typing import Annotated
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Route to ingest our resume / project files
@app.post("/rag-pinecone/files/")
async def ingest_file(
        file: UploadFile,
):
    loader = PyPDFLoader(file.filename)
    pages = loader.load_and_split()

    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    for page in pages:
        # Split data
        all_splits += text_splitter.split_documents(page)

    # Add to vectorDB
    # vectorstore = PineconeVectorStore.from_documents(
    #     documents=all_splits, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME
    # )

    return all_splits

# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)
add_routes(app, rag_pinecone_chain, path="/rag-pinecone")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

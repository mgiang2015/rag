from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_pinecone import chain as rag_pinecone_chain
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


app = FastAPI(redirect_slashes=False)

#################### API
@app.get("")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.post("/rag-pinecone/website")
async def ingest_website(url: str, status_code=200):
    # Ingest information from submitted URL
    # Load
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(url)
    data = loader.load()

    # Split
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # Add to vectorDB
    # Check if index already exists. If yes, delete index and create a new one
    from pinecone import Pinecone, PodSpec
    import os
    
    # configure client
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)
    spec = PodSpec(environment="gcp-starter")
    
    # check for and delete index if already exists
    index_name = os.environ.get("PINECONE_INDEX", "default")
    if index_name in pc.list_indexes().names():  
        pc.delete_index(index_name)
    
    # create a new index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002  
        metric='cosine',
        spec=spec  
    )

    from langchain_pinecone import PineconeVectorStore
    from langchain_community.embeddings import OpenAIEmbeddings
    vectorstore = PineconeVectorStore.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings(), index_name=index_name
    )

    return { "message" : "Successfully ingested article at " + url }

# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)
add_routes(app, rag_pinecone_chain, path="/rag-pinecone")

#################### Middleware to handle communication

# # handle CORS preflight requests
@app.options("/{full_path:path}")
async def preflight_handler(request: Request, full_path: str) -> Response:
    print("Handling preflight request")
    response = Response()
    if ("Origin" in request.headers):
        response.headers['Access-Control-Allow-Origin'] = str(request.headers["Origin"])
    else:
        response.headers['Access-Control-Allow-Origin'] = "*"
    
    response.headers['Access-Control-Max-Age'] = '1728000'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS,PUT,DELETE,PATCH'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization,Accept,Origin,DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Content-Range,Range'
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Length'] = '0'

    return response


# origins = ["http://localhost:3000",
#            "https://mgiang2015.github.io"]

# methods = ["POST", "GET", "OPTIONS", "PUT", "DELETE", "PATCH"]

# headers = ["Authorization","Accept","Origin","DNT","X-CustomHeader","Keep-Alive","User-Agent",
#            "X-Requested-With","If-Modified-Since","Cache-Control","Content-Type","Content-Range","Range"]

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

import os

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
    raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")

vectorstore = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# RAG without sources. Uncomment if no sources
template = """Answer the question based only on the following context. If you do not know, just say that you do not know. Be as thorough as possible:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input. Keep for no sources for now
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)

# Rag with sources. Uncomment if need sources
# prompt = hub.pull("rlm/rag-prompt")
# model = ChatOpenAI()

# rag_chain_from_docs = (
#     RunnablePassthrough.assign(context=(lambda x: x["context"]))
#     | prompt
#     | model
#     | StrOutputParser()
# )

# chain = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# ).assign(answer=rag_chain_from_docs)

# Rag with citation. Uncomment if need citations for each sentence
# from operator import itemgetter
# from typing import List
# from langchain_core.documents import Document

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You're a helpful AI assistant. Given a user question and some article snippets, answer the user question. For each point, cite the source and link to the source document. If none of the articles answer the question, just say you don't know.\n\nHere are the articles:{context}",
#         ),
#         ("human", "{question}"),
#     ]
# )
# model = ChatOpenAI()

# # Write tool for model to call!!
# from langchain_core.pydantic_v1 import BaseModel, Field

# class cited_answer(BaseModel):
#     """Answer the user question based only on the given sources, and cite the sources used."""

#     answer: str = Field(
#         ...,
#         description="The answer to the user question, which is based only on the given sources.",
#     )
#     citations: List[int] = Field(
#         ...,
#         description="The integer IDs of the SPECIFIC sources which justify the answer.",
#     )

# model_with_tool = model.bind_tools(
#     [cited_answer], # function to be called
#     tool_choice="cited_answer" # Can it be other things?
# )

# # Output parser for tool output
# from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
# output_parser = JsonOutputKeyToolsParser(key_name="cited_answer", return_single=True)

# def format_docs_with_id(docs: List[Document]) -> str:
#     formatted = [
#         f"Source ID: {i}\nArticle Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
#         for i, doc in enumerate(docs)
#     ]
#     return "\n\n" + "\n\n".join(formatted)

# format = itemgetter("docs") | RunnableLambda(format_docs_with_id)
# # subchain for generating an answer once we've done retrieval
# answer = prompt | model_with_tool | output_parser
# # complete chain that calls wiki -> formats docs to string -> runs answer subchain -> returns just the answer and retrieved docs.
# chain = (
#     RunnableParallel(question=RunnablePassthrough(), docs=retriever)
#     .assign(context=format)
#     .assign(answer=answer)
# )

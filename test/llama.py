

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from fastapi import FastAPI

app = FastAPI()


documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

@app.get("/")
async def main(): 
    return {"message": "Hello World"}

@app.post("/query")
async def root(query: str):
    response = query_engine.query(query)
    print(response)
    return {"message": response}
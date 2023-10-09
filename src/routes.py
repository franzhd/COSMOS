import os
from src.agent import Agent
from fastapi import FastAPI, Body, Request
from fastapi import File, UploadFile, Form
from fastapi.responses import HTMLResponse
import shutil
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from src.ingest import main as ingest_docs
from src.run_localGPT import ChatLlama2
import torch

def empty_uploaded_files():
    for filename in os.listdir("uploaded_files"):
            file_path = os.path.join("uploaded_files", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            else:
                continue
    for filename in os.listdir("src/SOURCE_DOCUMENTS"):
        file_path = os.path.join("src/SOURCE_DOCUMENTS", filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        else:
            continue


empty_uploaded_files()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to be specific here in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = Agent(dummy=True)

llama = ChatLlama2(dummy=True)

@app.get("/")
def main():
    content = """
        <body>
        <form action="/upload/" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
        </form>
        </body>
    """
    return HTMLResponse(content=content)

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     file_path = os.path.join("uploaded_files", file.filename)
    
#     print("_"*20)

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     agent.ingest(file_path)

#     # description = agent.summarize_documents(str(os.path.basename(os.path.basename(file_path))).split(".")[0])
#     # print(description + "_"*20 + "\n")

#     description = "AHH SONO LA description!"

#     return {"filename": file_path,
#             "description": description}

@app.post("/gpt/upload")
async def upload_files(files: List[UploadFile] = File(...)):

    uploaded_files = []

    for file in files:
        file_path = os.path.join("uploaded_files", file.filename)
        # print("_"*20)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        global agent
        agent = Agent()
        agent.ingest(file_path)
        description = "Ciao, sono Silvio  " + file.filename
        # description = agent.summarize_documents(str(os.path.basename(os.path.basename(file_path))).split(".")[0])
        # print(description + "_"*20 + "\n")
        
        uploaded_files.append({
            "filename": file.filename,
            "description": description
        })
    
    return {
        "uploaded_files": uploaded_files,
    }

@app.post("/llama/upload")
async def upload_files(files: List[UploadFile] = File(...)):

    from numba import cuda 
    device = cuda.get_current_device()
    device.reset()
    uploaded_files = []

    for file in files:
        file_path = os.path.join("src/SOURCE_DOCUMENTS", file.filename)
        # print("_"*20)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        


        description = "Ciao, sono Silvio  " + file.filename
        # description = agent.summarize_documents(str(os.path.basename(os.path.basename(file_path))).split(".")[0])
        # print(description + "_"*20 + "\n")
        
        uploaded_files.append({
            "filename": file.filename,
            "description": description
        })
    ingest_docs()
    global llama
    llama = ChatLlama2()
    return {
        "uploaded_files": uploaded_files,
    }


@app.post("/gpt/query")
async def root(request: Request):

    data = await request.json()
    query = data.get("text")
    global agent
    response = agent.ask(query)
    print(response)
    return {"text": response}

@app.post("/llama/query")

async def root(request: Request):
    from numba import cuda 
    device = cuda.get_current_device()
    device.reset()
    data = await request.json()
    query = data.get("text")
    global llama
    response = llama.ask(query)
    print(response)
    return {"text": response}

@app.post("/switch")
async def root(request: Request):
    data = await request.json()
    to_model = data.get("text")
    if to_model == 'gpt':
        global llama
        del llama
        torch.cuda.empty_cache()
        from numba import cuda 
        device = cuda.get_current_device()
        device.reset()
    elif to_model == 'llama':
        del agent
        from numba import cuda 
        device = cuda.get_current_device()
        device.reset()

        
# @app.post("/query")
# async def root(query: str):
#     response = agent.ask(query)
#     print(response)
#     return {"message": response}

# @app.post("/delete")
# async def delete_file(filename: str):
#     file_path = os.path.join("uploaded_files", filename)
#     os.remove(file_path)
#     return {"message": "File {filename} deleted"}

# @app.post("/delete")
# async def delete_file(request: Request):
#     data = await request.json()
#     filename = data.get("filename")

#     file_path = os.path.join("uploaded_files", filename)

#     os.remove(file_path)
#     return {"message": f"File {filename} deleted"}

# @app.post("/llama/delete")
# async def delete_file(request: Request):
#     data = await request.json()
#     filename = data.get("filename")

#     file_path = os.path.join("SOURCE_DOCUMENTS", filename)

#     os.remove(file_path)
#     return {"message": f"File {filename} deleted"}


@app.post(f'/llama/delete_all')
async def delete_file():
    empty_uploaded_files()
    return {"message": "All files deleted"}

@app.post(f'/gpt/delete_all')
async def delete_file():
    empty_uploaded_files()
    return {"message": "All files deleted"}

# @app.post("/llama/delete_all")
# async def delete_file():
#     empty_uploaded_files()
#     return {"message": "All files deleted"}

@app.get("/reset")
def reset_agent():
    agent.forget()
    return {"message": "Agent reset"}

@app.get("/llama/reset")
def reset_agent():
    torch.cuda.empty_cache()
    global llama
    llama = ChatLlama2()
    return {"message": "Agent reset"}
import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# 加載環境變量
load_dotenv()

# 檢查必要的環境變量
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("請在 .env 文件中設置 OPENAI_API_KEY")

app = FastAPI()

class ChatQuery(BaseModel):
    query: str

class RAGAssistant:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.chat_chain = None
        self.chat_history = []
        
    def load_documents(self, file_paths: List[str]):
        """加載文檔並創建向量存儲"""
        documents = []
        for path in file_paths:
            loader = TextLoader(path)
            documents.extend(loader.load())
            
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings
        )
        
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.7),
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )
    
    def chat(self, query: str) -> Dict:
        """與用戶進行對話"""
        if not self.chat_chain:
            return {"response": "請先加載知識庫文件。"}
        
        result = self.chat_chain({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        
        return {
            "response": result["answer"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }

# 創建助手實例
assistant = RAGAssistant()

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上傳知識庫文件"""
    file_paths = []
    for file in files:
        file_path = f"data/{file.filename}"
        os.makedirs("data", exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        file_paths.append(file_path)
    
    assistant.load_documents(file_paths)
    return {"message": "文件上傳成功"}

@app.post("/chat")
async def chat(query: ChatQuery):
    """與助手對話"""
    return assistant.chat(query.query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

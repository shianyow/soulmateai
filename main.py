import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加載環境變量
load_dotenv()

# 檢查必要的環境變量
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("請在 .env 文件中設置 OPENAI_API_KEY")

app = FastAPI()

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載靜態文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

class ChatQuery(BaseModel):
    query: str

class RAGAssistant:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.chat_chain = None
        self.chat_history = []
        self.uploaded_files = []  # 存儲已上傳的文件名稱
        
        # 定義提示模板
        self.qa_template = """你是一個專業的心理健康助手。請使用繁體中文回答所有問題。

背景知識：
{context}

在回答時請遵循以下原則：
1. 始終使用繁體中文
2. 保持專業、同理心和關懷的態度
3. 將上傳的文件作為背景知識，而不是直接重複內容
4. 根據用戶的具體問題和情況，提供個性化的建議
5. 如果涉及專業術語，請提供解釋
6. 使用 Markdown 格式來組織內容，保持回答的條理性
7. 回答要簡潔有力，避免冗長
8. 仔細閱讀聊天歷史，避免重複之前的建議
9. 當用戶表示理解、感謝或結束對話的意圖（例如：「了解」、「謝謝」、「好的」等）時，請簡短回應「很高興能幫上忙」或「不客氣」，不要繼續給出新的建議或嘗試延續對話

用戶問題：{question}
聊天歷史：{chat_history}

請根據上述原則和聊天歷史提供回應："""
        
    def load_documents(self, file_paths: List[str]):
        """加載文檔並創建向量存儲"""
        try:
            documents = []
            allowed_extensions = ['.txt', '.md', '.json', '.pdf']
            
            for path in file_paths:
                ext = os.path.splitext(path)[1].lower()
                if ext not in allowed_extensions:
                    raise ValueError(f"不支持的文件格式：{ext}。僅支持：{', '.join(allowed_extensions)}")
                
                if ext == '.pdf':
                    loader = PyPDFLoader(path)
                else:
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
            
            # 創建對話鏈
            PROMPT = PromptTemplate(
                template=self.qa_template,
                input_variables=["context", "question", "chat_history"]
            )
            
            self.chat_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    model_name="gpt-4",
                    temperature=0.7,
                    model_kwargs={
                        "top_p": 0.9,
                        "presence_penalty": 0.6,  # 降低重複內容的可能性
                        "frequency_penalty": 0.6  # 鼓勵使用更多樣的詞彙
                    }
                ),
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}  # 限制檢索的文檔數量
                ),
                combine_docs_chain_kwargs={
                    "prompt": PROMPT,
                    "document_prompt": PromptTemplate(
                        input_variables=["page_content"],
                        template="{page_content}"
                    )
                },
                return_source_documents=True,
                verbose=True  # 啟用詳細日誌
            )
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def chat(self, query: str) -> Dict:
        """與用戶進行對話"""
        try:
            if not self.chat_chain:
                logger.warning("Chat chain not initialized")
                return {"response": "請先上傳知識庫文件。"}
            
            result = self.chat_chain({"question": query, "chat_history": self.chat_history})
            self.chat_history.append((query, result["answer"]))
            
            logger.info(f"Successfully processed query: {query[:50]}...")
            return {
                "response": result["answer"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except Exception as e:
            logger.error(f"Error processing chat: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# 創建助手實例
assistant = RAGAssistant()

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上傳知識庫文件"""
    try:
        saved_files = []
        file_names = []  # 存儲文件名
        for file in files:
            file_path = f"data/{file.filename}"
            os.makedirs("data", exist_ok=True)
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file_path)
            file_names.append(file.filename)
        
        success = assistant.load_documents(saved_files)
        if success:
            assistant.uploaded_files = file_names  # 更新已上傳的文件列表
            logger.info(f"Successfully uploaded and processed {len(files)} files")
            return {
                "message": "文件上傳成功",
                "uploaded_files": file_names
            }
        else:
            raise HTTPException(status_code=500, detail="文件處理失敗")
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/uploaded-files")
async def get_uploaded_files():
    """獲取已上傳的文件列表"""
    return {"files": assistant.uploaded_files}

@app.post("/chat")
async def chat(query: ChatQuery):
    """與助手對話"""
    try:
        return assistant.chat(query.query)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# SoulMate AI - 你的心靈小助手

這是一個基於 RAG (Retrieval-Augmented Generation) 技術的心靈諮詢助手。它可以根據提供的知識庫，為用戶提供個性化的回答和建議。

## 功能特點

- 支持自定義知識庫
- 使用向量數據庫進行高效檢索
- 結合 LLM 生成個性化回答
- 提供 API 接口方便集成

## 安裝步驟

1. 安裝依賴：
```bash
pip install -r requirements.txt
```

2. 設置環境變量：
創建 `.env` 文件並添加以下內容：
```
OPENAI_API_KEY=你的OpenAI API密鑰
```

3. 運行服務：
```bash
python main.py
```

## 使用方法

1. 將知識庫文件放入 `data` 目錄
2. 啟動服務
3. 通過 API 或命令行界面與助手對話

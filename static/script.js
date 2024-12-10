// DOM 元素
const fileInput = document.getElementById('fileInput');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const chatHistoryDiv = document.getElementById('chatHistory');

// 配置 marked
marked.setOptions({
    breaks: true,
    sanitize: false
});

// 添加載入動畫
function createLoadingElement() {
    const loading = document.createElement('div');
    loading.className = 'loading';
    loading.innerHTML = '<span></span><span></span><span></span>';
    return loading;
}

// 獲取已上傳的文件列表
async function getUploadedFiles() {
    try {
        const response = await fetch('/uploaded-files');
        const data = await response.json();
        const filesList = document.getElementById('uploadedFilesList');
        filesList.innerHTML = '';
        data.files.forEach(file => {
            const li = document.createElement('li');
            li.textContent = file;
            filesList.appendChild(li);
        });
    } catch (error) {
        console.error('Error fetching uploaded files:', error);
        showSystemMessage('無法獲取文件列表，請重新整理頁面');
    }
}

// 顯示系統消息
function showSystemMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system-message';
    messageDiv.textContent = message;
    chatHistoryDiv.appendChild(messageDiv);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
}

// 添加消息到聊天界面
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
    
    if (role === 'user') {
        messageDiv.textContent = content;
    } else {
        messageDiv.innerHTML = marked.parse(content);
    }
    
    chatHistoryDiv.appendChild(messageDiv);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
    return messageDiv;
}

// 發送消息
async function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') return;
    
    // 禁用輸入和發送按鈕
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    // 清空輸入框並添加用戶消息
    userInput.value = '';
    addMessage('user', message);
    
    // 添加載入動畫
    const loadingDiv = addMessage('assistant', '');
    const loading = createLoadingElement();
    loadingDiv.appendChild(loading);
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 移除載入動畫並添加回應
        loadingDiv.innerHTML = marked.parse(data.response);
        
    } catch (error) {
        console.error('Error:', error);
        loadingDiv.remove();
        showSystemMessage('發送消息失敗，請稍後重試');
    } finally {
        // 重新啟用輸入和發送按鈕
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// 處理文件上傳
fileInput.addEventListener('change', async function(e) {
    const files = e.target.files;
    if (files.length === 0) return;

    const uploadBtn = document.getElementById('uploadBtn');
    const originalText = uploadBtn.innerHTML;
    uploadBtn.innerHTML = `
        <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        上傳中...
    `;
    uploadBtn.disabled = true;

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('上傳失敗');
        }

        showSystemMessage('文件上傳成功');
        await getUploadedFiles();
    } catch (error) {
        console.error('Error uploading files:', error);
        showSystemMessage('文件上傳失敗：' + error.message);
    } finally {
        uploadBtn.innerHTML = originalText;
        uploadBtn.disabled = false;
        fileInput.value = '';
    }
});

// 綁定上傳按鈕點擊事件
document.getElementById('uploadBtn').addEventListener('click', function() {
    fileInput.click();
});

// 發送按鈕點擊事件
sendBtn.addEventListener('click', sendMessage);

// 輸入框按下 Enter 鍵事件
userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 頁面加載時初始化
window.addEventListener('load', function() {
    addMessage('assistant', '你好！我是你的心靈小助手。請告訴我你想聊些什麼？');
    getUploadedFiles();
    userInput.focus();
});

# -*- coding: utf-8 -*-
"""
本地 RAG 网页版：浏览器里聊天问答，基于 data 文件夹文档
====================================================================
运行方式：在 PyCharm 终端执行  python web_app.py
启动后会自动打开浏览器页面，在输入框提问即可。

与 rag_simple.py 的区别：
  rag_simple.py = 命令行问答（用于调试）
  web_app.py    = 网页聊天界面（用于演示、交付给客户看）

【可改】= 可以按自己需要修改
【勿动】= 现阶段不要改，避免报错
"""

# ===================== 导入工具包（【勿动】）=====================
import os
import gradio as gr
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document


# ===================== 【可改】配置区 =====================
LLM_MODEL = "qwen2:1.5b"
EMBED_MODEL = "nomic-embed-text"
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
SUPPORTED_EXT = (".txt", ".pdf", ".docx")
REBUILD_INDEX = False

# 网页标题和说明
APP_TITLE = "本地知识库问答 Demo"
APP_DESC = "基于本地文档回答问题，资料里没有的内容会尽量说明不知道。"
# ========================================================


def read_one_file(path):
    """根据后缀读取单个文档，返回文本内容（【勿动】）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="gbk") as f:
                return f.read()
    elif ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def load_documents(folder):
    """读取文件夹下所有支持的文档，拼接成一段文本返回（【勿动】）。"""
    texts = []
    for name in os.listdir(folder):
        if name.lower().endswith(SUPPORTED_EXT):
            texts.append(read_one_file(os.path.join(folder, name)))
    return "\n".join(texts)


def build_or_load_vector_db(embedding):
    """有已保存的索引就直接加载；否则读文档新建并落盘（【勿动】）。"""
    cache_exists = os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0
    if cache_exists and not REBUILD_INDEX:
        print("检测到已保存的向量库，直接加载。")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

    print("读取文档并建立新的向量索引……")
    knowledge_text = load_documents(DATA_DIR)
    if not knowledge_text.strip():
        raise SystemExit("⚠️ data 文件夹里没有读到 .txt / .pdf / .docx 文档，请先放入再运行。")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_text(knowledge_text)
    print(f"共读取 {len(knowledge_text)} 个字符，切成 {len(chunks)} 个片段。")
    return Chroma.from_texts(chunks, embedding, persist_directory=CHROMA_DIR)


# ---------- 程序启动时初始化一次模型和检索器（【勿动】）----------
print("加载本地大模型：", LLM_MODEL)
llm = OllamaLLM(model=LLM_MODEL)

print("加载向量化模型：", EMBED_MODEL)
embedding = OllamaEmbeddings(model=EMBED_MODEL)

vector_db = build_or_load_vector_db(embedding)
retriever = vector_db.as_retriever()


def answer(question, history):
    """网页提问回调：检索资料 → 拼提示词 → 调用大模型（【勿动】）。"""
    if not question.strip():
        return "请输入问题。"

    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""基于下面的资料回答问题，资料里没有就说不知道，不要编造：
资料：{context}
问题：{question}
"""
    return llm.invoke(prompt)


# ---------- 搭建网页界面（【勿动】）----------
demo = gr.ChatInterface(
    fn=answer,
    title=APP_TITLE,
    description=APP_DESC,
    examples=["什么是 RAG？", "开源大模型是什么？", "Ollama 是做什么的？"],
    chatbot=gr.Chatbot(height=420),
)

if __name__ == "__main__":
    # 启动本地网页服务，自动打开浏览器；只允许本机访问
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False)

# -*- coding: utf-8 -*-
"""一次性链路测试：读取 data 文件夹（txt/pdf/docx），自动提问一次（非交互）。"""
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document

DATA_DIR = "./data"
SUPPORTED_EXT = (".txt", ".pdf", ".docx")


def read_one_file(path):
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
    texts = []
    for name in os.listdir(folder):
        if name.lower().endswith(SUPPORTED_EXT):
            texts.append(read_one_file(os.path.join(folder, name)))
    return "\n".join(texts)


knowledge_text = load_documents(DATA_DIR)
print("读取到字符数：", len(knowledge_text))

llm = OllamaLLM(model="qwen2:1.5b")
embedding = OllamaEmbeddings(model="nomic-embed-text")

chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_text(knowledge_text)
db = Chroma.from_texts(chunks, embedding)
retriever = db.as_retriever()

question = "这份文档里提到了哪些 AI 相关概念？"
docs = retriever.invoke(question)
context = "\n".join(d.page_content for d in docs)
prompt = f"基于资料回答问题，资料里没有就说不知道，不要编造：\n资料：{context}\n问题：{question}\n"
answer = llm.invoke(prompt)

print("=== 检索到的资料 ===")
print(context)
print("=== 模型回答 ===")
print(answer)

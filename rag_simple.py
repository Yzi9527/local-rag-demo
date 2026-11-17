# -*- coding: utf-8 -*-
"""
本地 RAG 演示：Ollama + LangChain + Chroma，读取本地文档问答（向量库已落盘）
====================================================================
现阶段你不需要自己写代码，只需要会改下面标注【可改】的几行。

运行前提（已完成）：
  1. 项目虚拟环境已装好依赖
  2. Ollama 已在后台运行
  3. 已拉取本地模型：
       ollama pull qwen2:1.5b          （对话模型）
       ollama pull nomic-embed-text    （向量化模型）

使用方法：
  把要提问的文档放进本项目的 data 文件夹，支持：.txt / .pdf / .docx
  注意：扫描版 PDF（图片拍照那种）提取不出文字，需要 OCR，本版本暂不支持。

关于向量库保存：
  第一次运行会读取文档、建立索引并保存到 chroma_db 文件夹；
  之后再次运行会直接加载已有索引，速度更快。
  如果你新增/修改了文档，把下面 REBUILD_INDEX 改成 True 重新建一次，
  或直接删掉 chroma_db 文件夹再运行。

【可改】= 可以按自己需要修改
【勿动】= 现阶段不要改，避免报错
"""

# ===================== 导入工具包（【勿动】）=====================
import os  # 用于读取文件夹里的文件
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader          # 读取 PDF
from docx import Document            # 读取 Word(.docx)


# ===================== 【可改】配置区 =====================
# 本地对话模型名称（对应 ollama list 里的名字）
LLM_MODEL = "qwen2:1.5b"

# 本地向量化模型名称
EMBED_MODEL = "nomic-embed-text"

# 知识库文件夹：把文档放在这里
DATA_DIR = "./data"

# 向量库保存文件夹（程序自动生成，不用手动建）
CHROMA_DIR = "./chroma_db"

# 支持读取的文档后缀
SUPPORTED_EXT = (".txt", ".pdf", ".docx")

# 是否强制重建索引：False=有缓存就直接用；True=重新读文档建索引
# 新增/修改文档后，把这里改成 True 跑一次即可
REBUILD_INDEX = False
# ========================================================


def read_one_file(path):
    """根据后缀读取单个文档，返回文本内容（【勿动】）。"""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # 个别 Windows 记事本存的是 GBK 编码，换 GBK 再读一次
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
    # 判断缓存索引是否存在：文件夹存在且里面有内容
    cache_exists = (
        os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0
    )

    if cache_exists and not REBUILD_INDEX:
        print("      检测到已保存的向量库，直接加载（跳过重新建索引）。")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding,
        )

    # 没有缓存或要求重建：读取文档 → 切片 → 向量化 → 落盘保存
    print("      读取文档并建立新的向量索引……")
    knowledge_text = load_documents(DATA_DIR)
    if not knowledge_text.strip():
        print("⚠️ data 文件夹里没有读到 .txt / .pdf / .docx 文档，请先放入再运行。")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_text(knowledge_text)
    print(f"      共读取 {len(knowledge_text)} 个字符，切成 {len(chunks)} 个片段。")

    return Chroma.from_texts(
        chunks,
        embedding,
        persist_directory=CHROMA_DIR,
    )


def main():
    # 1. 初始化本地对话大模型
    print("[1/4] 加载本地大模型：", LLM_MODEL)
    llm = OllamaLLM(model=LLM_MODEL)

    # 2. 初始化本地向量化模型
    print("[2/4] 加载向量化模型：", EMBED_MODEL)
    embedding = OllamaEmbeddings(model=EMBED_MODEL)

    # 3. 加载或新建向量库
    print("[3/4] 准备知识库索引……")
    vector_db = build_or_load_vector_db(embedding)
    if vector_db is None:
        return
    retriever = vector_db.as_retriever()

    # 4. 进入问答循环
    print("[4/4] RAG Demo 启动成功！输入问题，输入 quit 退出。")
    print("=" * 50)
    while True:
        question = input("\n你：")
        if question.strip().lower() == "quit":
            break

        # 4.1 先从知识库检索最相关的片段
        docs = retriever.invoke(question)
        context = "\n".join([d.page_content for d in docs])

        # 4.2 把检索到的资料 + 问题拼成提示词，交给大模型回答
        prompt = f"""基于下面的资料回答问题，资料里没有就说不知道，不要编造：
资料：{context}
问题：{question}
"""
        answer = llm.invoke(prompt)
        print(f"AI：{answer}")


if __name__ == "__main__":
    main()

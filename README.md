# 本地知识库问答 RAG Demo

基于 **Ollama 本地大模型 + LangChain + Chroma** 的最简检索增强生成（RAG）示例：把本地文档（TXT / PDF / Word）接入知识库，在命令行或网页界面中针对文档内容提问，模型基于资料回答，减少凭空编造。

> 本项目用于学习 RAG 完整链路与本地模型部署，不提供从零定制开发，仅作模板部署与调试参考。

## 功能特性

- 📄 读取本地 `.txt` / `.pdf` / `.docx` 文档作为知识库
- 🔒 全程本地运行：对话模型、向量化模型均通过 Ollama 在本机运行，不依赖外部在线接口
- 🗂️ 向量库落盘保存：首次运行建立索引，之后直接加载，无需重复处理文档
- 💬 两种交互方式：命令行问答（`rag_simple.py`）+ 网页聊天界面（`web_app.py`）
- 🧩 提示词约束：要求模型仅依据资料回答，资料没有就说明不知道

## 技术栈

| 环节 | 技术 / 工具 |
|---|---|
| 本地模型管理 | Ollama |
| 对话模型 | qwen2:1.5b（可在配置区更换） |
| 向量化模型 | nomic-embed-text |
| 应用编排框架 | LangChain |
| 向量数据库 | Chroma |
| 文档读取 | pypdf（PDF）、python-docx（Word） |
| 网页界面 | Gradio |
| 运行环境 | Python 3.10 / Windows |

## 项目结构

```
reg-demo/
├── rag_simple.py      # 命令行问答版（调试用）
├── web_app.py         # 网页聊天界面版（演示/交付用）
├── test_rag.py        # 一次性链路测试脚本
├── requirements.txt   # Python 依赖清单
├── .gitignore         # Git 忽略规则
├── data/              # 知识库文档放这里（.txt/.pdf/.docx）
│   └── knowledge.txt  # 示例文档
└── chroma_db/         # 向量库缓存（自动生成，不纳入 Git）
```

## 环境要求

- Windows 10 / 11
- Python 3.10（建议 3.10.x，避免高版本依赖兼容问题）
- 已安装 [Ollama](https://ollama.com/) 并在后台运行
- 电脑内存建议 8GB 以上；1.5b 模型可 CPU 运行，更大模型建议独立显卡

## 快速开始

### 1. 拉取本地模型

安装 Ollama 后，在命令行执行：

```bash
ollama pull qwen2:1.5b
ollama pull nomic-embed-text
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
# Windows 激活：
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. 放入知识库文档

把要提问的 `.txt` / `.pdf` / `.docx` 文档复制到 `data/` 文件夹。

> 注意：扫描版 / 图片型 PDF 提取不出文字，本版本暂不支持 OCR。

### 4. 运行

**命令行版：**

```bash
python rag_simple.py
```

**网页版：**

```bash
python web_app.py
```

启动后浏览器访问 `http://127.0.0.1:7860`（通常会自动打开），在输入框提问即可。

## 使用说明

### 更换 / 新增文档后重建索引

向量库保存在 `chroma_db/`，新增或修改文档后需要重建索引，二选一：

1. 把代码配置区的 `REBUILD_INDEX` 改为 `True`，运行一次后再改回 `False`；
2. 直接删除 `chroma_db/` 文件夹，重新运行程序自动重建。

> 更换向量化模型后，旧索引不通用，必须重建。

### 可配置项（代码配置区）

| 配置项 | 作用 |
|---|---|
| `LLM_MODEL` | 对话模型名称 |
| `EMBED_MODEL` | 向量化模型名称 |
| `DATA_DIR` | 知识库文档文件夹 |
| `CHROMA_DIR` | 向量库保存文件夹 |
| `REBUILD_INDEX` | 是否强制重建索引 |

## 工作原理（数据流）

1. 读取 `data/` 文档，切分成小段；
2. 向量化模型把每个片段转成向量，存入 Chroma 向量库；
3. 提问时，把问题也转成向量，检索出最相关的片段；
4. 把检索到的片段和问题拼成提示词，交给本地大模型生成回答。

## 常见问题

- **端口 7860 被占用**：停止其他占用程序，或在 `web_app.py` 中修改 `server_port`。
- **模型回答不准 / 编造**：1.5b 小模型能力有限，可在电脑性能允许时更换更大的对话模型（如 `qwen2:7b`），并确保文档内容与问题相关。
- **PDF 读不出文字**：多为扫描件，需要 OCR 预处理。
- **启动报 `ModuleNotFoundError`**：确认已激活虚拟环境并安装 `requirements.txt`。

## 可扩展方向

- 接入更多文档格式（Markdown、HTML、表格等）
- 对话历史记忆与多轮上下文
- 检索结果来源引用（回答中标注出自哪份文档）
- 部署到云服务器，支持外部访问
- 更换更大参数模型或接入多模态模型

## 备注

- 本项目为本地学习与演示用途；对外交付前需根据客户数据合规、模型商用许可等要求另行评估。
- 示例文档 `data/knowledge.txt` 可自行替换。

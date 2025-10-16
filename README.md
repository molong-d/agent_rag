# RAG项目：基于本地知识库的智能问答系统

本项目是一个在Windows 10环境下运行的基于RAG（检索增强生成）技术构建的AI代理系统。它使用本地部署的模型和数据库，实现了文档加载、分块、向量化存储和智能问答功能。

## 项目优化亮点

经过全面优化，本项目现在具有以下增强特性：
- 🚀 **性能优化**：使用多进程并行处理文档，显著提升加载速度
- 🚀 **流式输出**：答案实时生成，提供更流畅的用户体验
- 🚀 **会话记忆**：支持上下文对话，记住用户之前的交互
- 🚀 **智能缓存**：频繁问题快速响应，节省计算资源
- 🚀 **扩展文档支持**：新增Word、Excel、CSV、PowerPoint等格式支持
- 🚀 **完善的错误处理**：友好的错误提示和详细的日志记录
- 🚀 **进度显示**：操作过程可视化，清晰了解处理状态
- 🚀 **交互式问答**：新增交互式命令行界面，方便用户持续提问

## 项目结构

```
rag_project/
├── config.py           # 配置文件，管理路径和参数
├── rag_init.py         # 知识库初始化脚本
├── agent.py            # 智能问答代理脚本
├── test_environment.py # 环境测试脚本
├── requirements.txt    # 项目依赖
├── Dockerfile          # 容器化配置
└── README.md           # 项目说明文档
```

## 技术栈

- **框架**: LangChain 0.2.0
- **向量数据库**: Chroma
- **嵌入模型**: BAAI/bge-base-zh-v1.5（本地部署）
- **LLM**: Ollama + qwen3:4b
- **文档处理**: PyPDF、TextLoader等

## 功能特点

1. **扩展文档支持**：支持PDF、TXT、Markdown、Word(.docx)、Excel(.xlsx)、CSV和PowerPoint(.pptx)格式
2. **并行处理**：多进程并行加载和处理文档，大幅提升性能
3. **智能分块**：针对中文文本优化的分块策略，保留语义完整性
4. **本地存储**：使用本地Chroma数据库存储向量数据
5. **会话记忆**：支持上下文感知的多轮对话
6. **智能缓存**：缓存常见问题答案，提升响应速度
7. **交互式界面**：新增命令行交互式问答模式
8. **流式输出**：答案实时生成，提升用户体验
9. **详细引用**：显示答案来源文档，增强可信度
10. **错误处理**：完善的日志记录和友好的错误提示机制

## 安装步骤

### 1. 环境准备

- Python 3.8+
- Ollama（已安装qwen3:4b模型）
- 足够的磁盘空间存储模型和数据

### 2. 安装依赖

```bash
# 进入项目目录
cd g:\rag_project

# 安装Python依赖
pip install -r requirements.txt
```

### 3. 准备数据和模型

确保以下路径存在且包含相应文件：

- `G:/rag_data/knowledge_docs/` - 存放要处理的文档
- `G:/rag_data/models/bge-base-zh-v1.5/` - 存放嵌入模型（如果不存在，会自动下载）
- `G:/rag_data/chroma_data/` - 向量数据库存储路径（会自动创建）

## 配置说明

项目配置集中在 `config.py` 文件中：

```python
# 路径配置
PATHS = {
    "knowledge_docs": "G:/rag_data/knowledge_docs",
    "chroma_data": "G:/rag_data/chroma_data",
    "model_path": "G:/rag_data/models/bge-base-zh-v1.5"
}

# 模型配置
MODEL_CONFIG = {
    "embedding_model": "BAAI/bge-base-zh-v1.5",
    "ollama_model": "qwen3:4b",
    "temperature": 0.3
}

# 其他配置项...
```

根据你的环境修改相应的路径和参数。

## 使用方法

### 1. 测试环境

首先运行环境测试脚本，检查所有组件是否正常工作：

```bash
python test_environment.py
```

### 2. 初始化知识库

将文档转换为向量并存储到数据库中：

```bash
python rag_init.py
```

此脚本现在会：
- 并行处理多种格式的文档
- 智能分块并清理内容
- 分批向量化并存储到数据库
- 显示详细的进度信息

### 3. 运行问答系统

#### 3.1 交互式问答模式（推荐）

```bash
python agent.py --interactive
```

在交互式模式下，您可以：
- 输入问题获取答案
- 输入 `exit` 或 `quit` 退出系统
- 输入 `clear` 清空缓存
- 系统会自动记住对话上下文

#### 3.2 单次问答模式

```bash
python agent.py  --interactive
```

系统会使用 `agent.py` 的 `main()` 函数中预设的问题进行测试，并以流式方式输出答案。

## Docker部署（可选）

### 构建镜像

```bash
docker build -t rag_project .
```

### 运行容器

在Windows Docker Desktop中运行：

```bash
docker run -v G:/rag_data/knowledge_docs:/app/data/knowledge_docs -v G:/rag_data/chroma_data:/app/data/chroma_data -v G:/rag_data/models:/app/data/models rag_project
```

> 注意：运行Docker容器前，请修改`config.py`中的路径为容器内的路径格式：`/app/data/...`

## 常见问题

### 1. 文档加载失败

- 检查文档路径是否正确
- 确保安装了所有依赖：`pip install -r requirements.txt`
- 对于PDF文件，确保安装了PyPDF库
- 对于Word、Excel等新增格式，确保已安装对应依赖：python-docx、openpyxl等

### 2. 模型加载失败

- 确保Ollama已正确安装并运行
- 确保已下载qwen3:4b模型：`ollama pull qwen3:4b`
- 检查模型路径配置是否正确

### 3. 向量数据库连接问题

- 默认使用本地文件模式，不需要启动额外的Chroma服务
- 如果需要使用客户端-服务器模式，请修改`config.py`中的`CHROMA_CONFIG`

### 4. 如何提高系统响应速度？

- 确保启用缓存：在`config.py`中设置`CACHE_CONFIG["enabled"] = True`
- 增加缓存容量：提高`CACHE_CONFIG["max_size"]`的值
- 在有GPU的机器上，设置`MODEL_CONFIG["device"] = "cuda"`

### 5. 如何启用或禁用会话记忆？

- 在`config.py`中修改`CONVERSATION_CONFIG["memory_type"]`
- 设置为`"buffer"`启用会话记忆，设置为`None`禁用会话记忆
- 可以调整`CONVERSATION_CONFIG["max_history_length"]`控制历史消息数量

### 6. 如何优化检索质量？

- 调整`config.py`中的`CHUNK_CONFIG`参数，优化文本分块
- 在`RETRIEVAL_CONFIG`中尝试不同的`search_type`（similarity、mmr等）
- 调整`RETRIEVAL_CONFIG["k"]`值，控制检索的文档片段数量

## 日志文件

项目现在使用更灵活的日志系统：
- 日志文件保存在 `logs/` 目录
- 每个操作会话生成独立的日志文件，命名格式为 `agent_YYYYMMDD_HHMMSS.log` 或 `rag_init_YYYYMMDD_HHMMSS.log`
- 日志包含详细的执行步骤、错误信息和调试数据
- 可以在 `config.py` 中通过 `LOGGING_CONFIG` 调整日志级别和格式

## 许可证

MIT License
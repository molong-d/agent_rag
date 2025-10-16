# RAG项目配置文件
import os
import platform

# 获取当前系统信息，以便跨平台兼容
SYSTEM = platform.system()

# 路径配置 - 支持相对路径，提高可移植性
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "knowledge_docs": os.path.join(BASE_DIR, "rag_data", "knowledge_docs"),
    "chroma_data": os.path.join(BASE_DIR, "rag_data", "chroma_data"),
    "model_path": os.path.join(BASE_DIR, "rag_data", "models", "bge-base-zh-v1.5"),
    "cache_dir": os.path.join(BASE_DIR, "rag_data", "cache")
}

# 确保目录存在
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    "embedding_model": "BAAI/bge-base-zh-v1.5",
    # "ollama_model": "qwen3:4b",
    # "ollama_model": "gemma3:1b",
    # "ollama_model": "deepseek-r1:1.5b",
    "ollama_model": "deepseek-r1:7b",

    "temperature": 0.3,
    "max_tokens": 1024,  # 最大生成token数
    "device": "cpu"  # 设备选择: "cpu" 或 "cuda"
}

# 文本分块配置
CHUNK_CONFIG = {
    "chunk_size": 300,
    "chunk_overlap": 30,
    "separators": ["\n\n", "\n", "。", "，", " ", "；"],
    "keep_separator": True,  # 保留分隔符
    "add_start_index": True  # 添加起始索引
}

# 检索配置
RETRIEVAL_CONFIG = {
    "k": 3,  # 检索前k个最相关的文档
    "search_type": "similarity",  # 检索类型: "similarity", "mmr", "similarity_score_threshold"
    "search_kwargs": {
        # 如果使用similarity_score_threshold
        # "score_threshold": 0.7
    }
}

# Chroma配置
CHROMA_CONFIG = {
    "use_client": False,  # 使用本地文件模式而非客户端-服务器模式
    "client_settings": {
        "chroma_server_host": "localhost",
        "chroma_server_port": 8000
    } if False else None,  # 如果启用客户端模式，设置连接信息
    "collection_name": "documents",  # 集合名称
    "collection_metadata": {"hnsw:space": "cosine"}  # 使用余弦相似度
}

# 缓存配置
CACHE_CONFIG = {
    "enabled": True,
    "max_size": 100,  # 最大缓存条目数
    "ttl": 3600  # 缓存过期时间(秒)
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(module)s - %(message)s",  # 增加模块名
    "encoding": "utf-8",  # 使用UTF-8编码避免Windows下的编码问题
    "log_file": os.path.join(BASE_DIR, "rag_project", "logs")
}

# 确保日志目录存在
os.makedirs(LOGGING_CONFIG["log_file"], exist_ok=True)

# 对话配置
CONVERSATION_CONFIG = {
    "max_history_length": 5,  # 最大历史记录长度
    "memory_type": "conversation_buffer"  # 内存类型
}
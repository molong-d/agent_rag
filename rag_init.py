import os
import logging
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
    Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document

# 导入配置
from config import PATHS, MODEL_CONFIG, CHUNK_CONFIG, CHROMA_CONFIG, LOGGING_CONFIG

# 配置日志
log_file = os.path.join(LOGGING_CONFIG["log_file"], f"rag_init_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    encoding=LOGGING_CONFIG["encoding"],
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 支持的文档类型和对应的加载器
DOCUMENT_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader
}

# 添加进度显示函数
def show_progress(message, step=None, total=None):
    """显示进度信息，同时输出到控制台和日志"""
    if step and total:
        progress = f"[{step}/{total}] "
    else:
        progress = ""
    full_message = f"{progress}{message}"
    print(f"[进度] {full_message}")
    logger.info(full_message)

# 1. 加载本地文档
def load_documents():
    try:
        docs_dir = PATHS["knowledge_docs"]
        
        # 检查目录是否存在
        if not os.path.exists(docs_dir):
            raise FileNotFoundError(f"文档目录不存在: {docs_dir}")
        
        # 获取目录中的所有文件
        all_files = []
        for root, _, files in os.walk(docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in DOCUMENT_LOADERS:
                    all_files.append((file_path, file_ext))
        
        if not all_files:
            logger.warning(f"警告: 在目录 {docs_dir} 中未找到支持的文档类型")
            return []
        
        show_progress(f"找到 {len(all_files)} 个支持的文档文件")
        
        # 并行加载文档以提高效率
        all_docs = []
        max_workers = min(4, len(all_files))  # 根据文件数量调整线程数
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有加载任务
            future_to_file = {
                executor.submit(load_single_document, file_path, file_ext): file_path 
                for file_path, file_ext in all_files
            }
            
            # 处理结果
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_path = future_to_file[future]
                try:
                    docs = future.result()
                    if docs:
                        all_docs.extend(docs)
                        show_progress(f"已加载 {file_path}", step=i, total=len(all_files))
                except Exception as e:
                    logger.error(f"加载文件 {file_path} 时出错: {str(e)}")
        
        if all_docs:
            logger.info(f"成功加载 {len(all_docs)} 个文档")
        else:
            logger.warning("未成功加载任何文档")
            
        return all_docs
    
    except Exception as e:
        logger.error(f"加载文档过程中发生错误: {str(e)}")
        raise

def load_single_document(file_path, file_ext):
    """加载单个文档"""
    try:
        loader_class = DOCUMENT_LOADERS[file_ext]
        
        # 为不同加载器设置特定参数
        if file_ext == ".csv":
            loader = loader_class(file_path, encoding="utf-8")
        elif file_ext == ".txt":
            # 尝试多种编码以提高兼容性
            try:
                loader = loader_class(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8编码失败，尝试GBK编码: {file_path}")
                loader = loader_class(file_path, encoding="gbk")
        elif file_ext in [".xlsx", ".xls"]:
            try:
                loader = loader_class(file_path)
                if hasattr(loader, 'mode'):
                    loader.mode = 'text'  # 确保提取文本模式
            except Exception as load_err:
                logger.error(f"初始化加载器失败: {str(load_err)}")
                raise
        else:
            loader = loader_class(file_path)
        
        docs = loader.load()
        # 允许空内容，但添加警告日志
        logger.debug(f"文档加载成功，页数: {len(docs)}, 第一页内容长度: {len(docs[0].page_content.strip()) if docs else 0}")
        if docs and len(docs[0].page_content.strip()) < 10:
            logger.warning(f"文档内容极少: {file_path}")
        
        # 添加元数据
        for doc in docs:
            doc.metadata["source"] = file_path
            doc.metadata["file_type"] = file_ext
            doc.metadata["loaded_at"] = datetime.now().isoformat()
        
        return docs
    except Exception as e:
        logger.error(f"加载失败文件: {file_path}")
        logger.error(f"错误详情: {str(e)}")
        raise

# 2. 文档分块
def split_documents(documents):
    try:
        if not documents:
            logger.warning("没有文档可供分块")
            return []
        
        # 使用配置文件中的分块设置
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CONFIG["chunk_size"],
            chunk_overlap=CHUNK_CONFIG["chunk_overlap"],
            separators=['\n\n', '\n', '。', ' '],
            keep_separator=True,
            add_start_index=True
        )
        logger.debug(f"分块参数: 大小{CHUNK_CONFIG['chunk_size']} 重叠{CHUNK_CONFIG['chunk_overlap']}")
        
        # 分块处理
        show_progress("开始文档分块处理...")
        logger.debug(f"待分块文档示例:\n{documents[0].page_content[:500]}")
        splits = text_splitter.split_documents(documents)
        logger.info(f"初步分块数量: {len(splits)}")
        
        # 优化分块后的文档
        optimized_splits = []
        for i, split in enumerate(splits):
            # 清理空白
            raw_content = split.page_content
            content = raw_content.strip()
            logger.debug(f"分块 {i} 原始内容长度: {len(raw_content)}, 处理后长度: {len(content)}")
            
            # 降低过滤标准，允许更短的分块
            if len(content) > 10:  # 只过滤非常短的空白内容
                # 确保元数据完整
                metadata = split.metadata.copy()
                metadata["chunk_id"] = f"{metadata.get('source', 'unknown')}_chunk_{len(optimized_splits)}"
                metadata["chunk_size"] = len(content)
                
                optimized_split = Document(page_content=content, metadata=metadata)
                optimized_splits.append(optimized_split)
        
        if optimized_splits:
            show_progress(f"文档分块完成，共 {len(optimized_splits)} 块")
            logger.debug(f"第一块内容示例: {optimized_splits[0].page_content[:100]}...")
        else:
            logger.warning("未生成任何有效的分块")
            
        return optimized_splits
    except Exception as e:
        logger.error(f"文档分块过程中发生错误: {str(e)}")
        raise

# 3. 初始化向量数据库
def init_vector_db(splits):
    try:
        if not splits:
            raise ValueError("没有分块文本可供向量化")
        
        # 确保数据目录存在
        os.makedirs(PATHS["chroma_data"], exist_ok=True)
        
        # 加载本地嵌入模型
        show_progress(f"正在加载嵌入模型: {MODEL_CONFIG['embedding_model']}")
        
        # 检查模型路径是否存在
        if not os.path.exists(PATHS["model_path"]):
            logger.warning(f"警告: 模型缓存路径 {PATHS['model_path']} 不存在，将尝试下载")
        
        # 加载嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_CONFIG["embedding_model"],
            cache_folder=PATHS["model_path"],
            model_kwargs={'device': MODEL_CONFIG.get('device', 'cpu')},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        show_progress(f"模型加载成功（缓存路径：{PATHS['model_path']}")
        
        # 处理大型知识库时的分批操作
        batch_size = 1000
        total_batches = (len(splits) + batch_size - 1) // batch_size
        
        # 如果数据库已存在，直接清空（不备份）
        if os.path.exists(PATHS["chroma_data"]):
            logger.info(f"检测到现有知识库，正在清理: {PATHS['chroma_data']}")
            try:
                # 清空目录
                shutil.rmtree(PATHS["chroma_data"])
                os.makedirs(PATHS["chroma_data"], exist_ok=True)
            except Exception as e:
                logger.error(f"清理现有知识库时出错: {str(e)}")
                raise
        
        # 分批创建向量数据库
        db = None
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(splits))
            batch_splits = splits[start_idx:end_idx]
            
            show_progress(f"正在处理批次 {batch_idx + 1}/{total_batches}，包含 {len(batch_splits)} 个分块")
            
            # 根据配置选择创建模式
            if batch_idx == 0:
                # 第一批次创建数据库
                if CHROMA_CONFIG["use_client"] and CHROMA_CONFIG["client_settings"]:
                    # 客户端-服务器模式
                    db = Chroma.from_documents(
                        documents=batch_splits,
                        embedding=embeddings,
                        client_settings=CHROMA_CONFIG["client_settings"],
                        persist_directory=PATHS["chroma_data"],
                        collection_name=CHROMA_CONFIG.get("collection_name", "documents"),
                        collection_metadata=CHROMA_CONFIG.get("collection_metadata")
                    )
                    logger.info("使用客户端-服务器模式创建Chroma数据库")
                else:
                    # 本地文件模式
                    db = Chroma.from_documents(
                        documents=batch_splits,
                        embedding=embeddings,
                        persist_directory=PATHS["chroma_data"],
                        collection_name=CHROMA_CONFIG.get("collection_name", "documents"),
                        collection_metadata=CHROMA_CONFIG.get("collection_metadata")
                    )
                    logger.info("使用本地文件模式创建Chroma数据库")
            else:
                # 后续批次添加文档
                db.add_documents(batch_splits)
        
        # 不再需要显式调用persist()，Chroma 0.4.x+版本会自动持久化
        
        show_progress(f"知识库初始化完成！共处理 {len(splits)} 块，数据存于：{PATHS['chroma_data']}")
        logger.info(f"知识库初始化统计：")
        logger.info(f"- 总文档块数: {len(splits)}")
        logger.info(f"- 存储路径: {PATHS['chroma_data']}")
        
        # 验证知识库
        if db:
            collection = db.get()
            logger.info(f"验证结果：知识库包含 {len(collection['ids'])} 个向量")
        
        return db
    except Exception as e:
        logger.error(f"初始化向量数据库过程中发生错误: {str(e)}")
        raise

# 主函数
def main():
    try:
        show_progress("开始初始化RAG知识库...")
        start_time = datetime.now()
        
        # 1. 加载文档
        show_progress("[步骤1/3] 正在加载文档...")
        docs = load_documents()
        show_progress(f"已加载 {len(docs)} 个文档")
        
        # 2. 文档分块
        show_progress("[步骤2/3] 正在进行文档分块...")
        splits = split_documents(docs)
        show_progress(f"文档分块完成，共 {len(splits)} 块")
        
        # 3. 初始化向量数据库
        show_progress("[步骤3/3] 正在初始化向量数据库...")
        db = init_vector_db(splits)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        show_progress(f"RAG知识库初始化成功完成！总用时：{duration:.2f}秒")
        
        return db
        
    except KeyboardInterrupt:
        logger.warning("操作被用户中断")
        print("\n操作已中断")
        return None
    except Exception as e:
        error_msg = f"RAG知识库初始化失败: {str(e)}"
        show_progress(f"错误: {error_msg}")
        logger.error(error_msg, exc_info=True)
        raise

# 程序入口
if __name__ == "__main__":
    main()

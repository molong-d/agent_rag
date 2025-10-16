# 1. 移除 ChromaClientSettings 导入（旧版本不支持）
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 加载文档（逻辑不变）
def load_documents():
    pdf_loader = DirectoryLoader(path="G:/rag_data/knowledge_docs", glob="*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(path="G:/rag_data/knowledge_docs", glob="*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(path="G:/rag_data/knowledge_docs", glob="*.md", loader_cls=UnstructuredMarkdownLoader)
    return pdf_loader.load() + txt_loader.load() + md_loader.load()

# 文本分块（逻辑不变）
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "，", " ", "；"]
    )
    splits = text_splitter.split_documents(documents)
    print(f"📝 分块详情：共 {len(splits)} 块，前1块内容：{splits[0].page_content[:100]}..." if splits else "❌ 未生成任何分块")
    return splits

# 2. 向量化存储（核心修正：用字典传连接配置，persist_directory 独立传递）
def init_vector_db(splits):
    # 加载本地模型（路径确保正确）
    embeddings = HuggingFaceEmbeddings(
        model_name="G:/rag_data/models/bge-base-zh-v1.5",  # 本地模型路径
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 修正：用字典传递连接配置（仅包含主机和端口，旧版本支持）
    client_settings = {
        "chroma_server_host": "localhost",  # 本地Chroma容器地址
        "chroma_server_port": 8000          # 容器端口
    }
    
    # persist_directory 作为独立参数，不放在 client_settings 中
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        client_settings=client_settings,  # 字典格式的连接配置
        persist_directory="G:/rag_data/chroma_data"  # 本地数据存储路径（独立参数）
    )
    db.persist()
    print(f"✅ 知识库初始化完成！共处理 {len(splits)} 个文档片段，数据已存入 G:\\rag_data\\chroma_data")

if __name__ == "__main__":
    docs = load_documents()
    print(f"📄 已加载 {len(docs)} 个文档")
    splits = split_documents(docs)
    print(f"✂️ 文档分块完成，共 {len(splits)} 块")
    init_vector_db(splits)
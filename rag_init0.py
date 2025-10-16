# 导入正确的模块（从langchain_community导入）
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载文档（分类型加载，避免type_loader_map参数）
def load_documents():
    # 加载PDF文件
    pdf_loader = DirectoryLoader(
        path="G:/rag_data/knowledge_docs",  # 本地文档目录
        glob="*.pdf",  # 只匹配PDF文件
        loader_cls=PyPDFLoader  # 指定PDF加载器
    )
    
    # 加载TXT文件
    txt_loader = DirectoryLoader(
        path="G:/rag_data/knowledge_docs",
        glob="*.txt",  # 匹配TXT文件
        loader_cls=TextLoader  # 指定TXT加载器
    )
    
    # 加载Markdown文件
    md_loader = DirectoryLoader(
        path="G:/rag_data/knowledge_docs",
        glob="*.md",  # 匹配MD文件
        loader_cls=UnstructuredMarkdownLoader  # 指定MD加载器
    )
    
    # 合并所有文档
    pdf_docs = pdf_loader.load()
    txt_docs = txt_loader.load()
    md_docs = md_loader.load()
    return pdf_docs + txt_docs + md_docs  # 合并为一个列表

# 2. 文本分块（逻辑不变）
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    return text_splitter.split_documents(documents)

# 3. 向量化并存储到Chroma
def init_vector_db(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    # 连接本地Chroma容器
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        client_settings={"chroma_server_host": "localhost", "chroma_server_port": 8000},
        persist_directory="G:/rag_data/chroma_data"
    )
    db.persist()
    print(f"✅ 知识库初始化完成！共处理 {len(splits)} 个文档片段，数据已存入 G:\\rag_data\\chroma_data")

if __name__ == "__main__":
    docs = load_documents()
    print(f"📄 已加载 {len(docs)} 个文档")
    splits = split_documents(docs)
    print(f"✂️ 文档分块完成，共 {len(splits)} 块")
    init_vector_db(splits)
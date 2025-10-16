import os
import time
import logging
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 导入配置
from config import PATHS, MODEL_CONFIG, RETRIEVAL_CONFIG, CHROMA_CONFIG, LOGGING_CONFIG, CACHE_CONFIG, CONVERSATION_CONFIG

# 配置日志
log_file = os.path.join(LOGGING_CONFIG["log_file"], f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

# 缓存系统
class SimpleCache:
    """简单的内存缓存系统"""
    def __init__(self, max_size=100, ttl=3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        
    def _generate_key(self, question):
        """生成缓存键"""
        return hashlib.md5(question.encode()).hexdigest()
    
    def get(self, question):
        """获取缓存值"""
        key = self._generate_key(question)
        if key in self.cache:
            value, timestamp = self.cache[key]
            # 检查是否过期
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                logger.debug(f"缓存命中: {question[:30]}...")
                return value
            else:
                # 过期删除
                del self.cache[key]
                logger.debug(f"缓存过期: {question[:30]}...")
        return None
    
    def set(self, question, answer):
        """设置缓存值"""
        # 如果缓存已满，删除最早的项目
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"缓存已满，删除最早项")
        
        key = self._generate_key(question)
        self.cache[key] = (answer, datetime.now())
        logger.debug(f"缓存设置: {question[:30]}...")

# 创建缓存实例
if CACHE_CONFIG["enabled"]:
    cache = SimpleCache(max_size=CACHE_CONFIG["max_size"], ttl=CACHE_CONFIG["ttl"])
else:
    cache = None

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

# 自定义提示模板
DEFAULT_PROMPT_TEMPLATE = """
【指令】
你是一个严格遵循文档的RAG问答系统，请按以下步骤处理问题：

1. 分析阶段：
   - 识别问题类型（事实查询/逻辑推理/无关问题）
   - 检查文档相关性（完全相关/部分相关/不相关）

2. 回答规则：
   - 当文档完全覆盖问题时：综合多个文档片段给出精准回答
   - 当文档部分相关时：
     * 只基于文档内容回答已知部分
     * 明确说明文档的局限处
   - 当文档不相关时：严格拒绝回答

3. 格式要求：
   - 使用【已知信息】、【文档依据】等标记说明来源
   - 使用中文书面语，保持专业但易懂
   - 分点论述复杂问题，重要信息优先

【文档内容】
{context}

【待回答问题】
{question}

【系统响应】
"""

# 初始化嵌入模型
@lru_cache(maxsize=1)
def initialize_embeddings():
    """初始化嵌入模型（使用lru_cache缓存实例）"""
    try:
        show_progress(f"开始加载嵌入模型: {MODEL_CONFIG['embedding_model']}")
        
        # 检查模型路径是否存在
        if not os.path.exists(PATHS["model_path"]):
            logger.warning(f"警告: 模型缓存路径 {PATHS['model_path']} 不存在，将尝试下载")
        
        # 加载本地嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_CONFIG["embedding_model"],
            cache_folder=PATHS["model_path"],
            model_kwargs={'device': MODEL_CONFIG.get('device', 'cpu')},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        show_progress("嵌入模型加载成功")
        return embeddings
    except Exception as e:
        logger.error(f"加载嵌入模型时发生错误: {str(e)}", exc_info=True)
        raise

# 向量数据库相关函数
def load_vector_db(embeddings):
    """加载向量数据库"""
    try:
        show_progress(f"开始加载知识库: {PATHS['chroma_data']}")
        
        # 检查数据库路径是否存在
        if not os.path.exists(PATHS["chroma_data"]):
            raise FileNotFoundError(f"知识库路径不存在: {PATHS['chroma_data']}\n请先运行rag_init.py初始化知识库")
        
        # 使用更兼容的方式加载Chroma数据库
        db = Chroma(
            persist_directory=PATHS["chroma_data"],
            embedding_function=embeddings,
            collection_name=CHROMA_CONFIG.get("collection_name", "documents")
        )
        
        # 尝试获取集合内容，使用更灵活的检查方式
        try:
            # 先尝试直接获取文档
            documents = db.get()
            if not documents.get('ids'):
                # 可能是集合名称问题，尝试使用默认集合
                db = Chroma(
                    persist_directory=PATHS["chroma_data"],
                    embedding_function=embeddings
                )
                documents = db.get()
                
            if documents.get('ids'):
                show_progress(f"知识库加载成功，包含 {len(documents['ids'])} 个向量")
                return db
            else:
                raise ValueError(f"知识库为空: {PATHS['chroma_data']}\n请检查文档是否正确加载")
        except Exception as inner_e:
            logger.warning(f"直接检查集合时出错: {str(inner_e)}，尝试替代方法")
            # 替代方法：尝试查询操作验证数据库
            try:
                # 尝试一个简单查询来验证数据库
                results = db.similarity_search("测试查询", k=1)
                if results:
                    show_progress(f"知识库加载成功")
                    return db
                else:
                    raise ValueError(f"知识库为空或无法正常查询: {PATHS['chroma_data']}")
            except Exception as query_e:
                logger.error(f"查询验证失败: {str(query_e)}")
                raise ValueError(f"加载知识库失败: {PATHS['chroma_data']}\n详细错误: {str(query_e)}")
    except Exception as e:
        logger.error(f"加载知识库时发生错误: {str(e)}", exc_info=True)
        raise

# 初始化语言模型
@lru_cache(maxsize=1)
def initialize_llm():
    """初始化语言模型（使用lru_cache缓存实例）"""
    try:
        show_progress(f"开始初始化LLM: {MODEL_CONFIG['ollama_model']}")
        
        # 加载 Ollama 模型
        llm = Ollama(
            model=MODEL_CONFIG["ollama_model"],
            temperature=MODEL_CONFIG["temperature"],
            num_predict=MODEL_CONFIG.get("max_tokens", 1024),
            # 添加超时设置
            # timeout=60.0
            timeout=300.0

        )
        
        # 测试模型连接
        try:
            # 使用一个简单的提示测试模型是否正常工作
            test_response = llm.invoke("请说一声'你好'确认连接正常")
            if "你好" in test_response:
                show_progress("LLM初始化成功并连接正常")
            else:
                logger.warning(f"LLM测试响应异常: {test_response}")
        except Exception as test_error:
            logger.warning(f"LLM连接测试失败，但将继续尝试: {str(test_error)}")
        
        return llm
    except Exception as e:
        logger.error(f"初始化LLM时发生错误: {str(e)}", exc_info=True)
        raise

# 创建会话内存
def create_conversation_memory():
    """创建会话内存"""
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="query",  # 修改为与prompt模板一致的键名
            output_key="result",
            k=CONVERSATION_CONFIG["max_history_length"]
        )
        show_progress(f"会话内存创建成功，最大历史记录长度: {CONVERSATION_CONFIG['max_history_length']}")
        return memory
    except Exception as e:
        logger.error(f"创建会话内存时发生错误: {str(e)}", exc_info=True)
        # 返回None表示使用无会话模式
        return None

# 创建问答链
def create_qa_chain(db, llm, memory=None):
    """创建问答链"""
    try:
        show_progress("开始创建问答链")
        
        # 创建检索器
        retriever = db.as_retriever(
            search_type=RETRIEVAL_CONFIG["search_type"],
            search_kwargs={
                "k": RETRIEVAL_CONFIG["k"],
                **RETRIEVAL_CONFIG.get("search_kwargs", {})
            }
        )
        
        # 创建提示模板
        prompt = PromptTemplate(
            template=DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # 构建 QA 链
        chain_kwargs = {
            "llm": llm,
            "chain_type": "stuff",
            "retriever": retriever,
            "return_source_documents": True,
            "chain_type_kwargs": {
                "prompt": prompt
            }
        }
        
        # 如果提供了内存，添加到链中
        if memory:
            chain_kwargs["memory"] = memory
        
        qa_chain = RetrievalQA.from_chain_type(**chain_kwargs)
        
        show_progress("问答链创建成功")
        return qa_chain
    except Exception as e:
        logger.error(f"创建问答链时发生错误: {str(e)}", exc_info=True)
        raise

# 执行文档检索
def perform_retrieval(retriever, question):
    """执行文档检索"""
    try:
        show_progress(f"开始检索相关文档，问题: {question[:50]}...")
        
        # 执行检索
        relevant_docs = retriever.get_relevant_documents(question)
        show_progress(f"检索到 {len(relevant_docs)} 个相关文档片段")
        
        # 记录检索结果
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get("source", "未知来源")
            logger.debug(f"第 {i} 个片段来源: {source}，内容前100字符: {doc.page_content[:100]}...")
        
        if not relevant_docs:
            logger.warning("警告: 未检索到任何相关文档")
        
        return relevant_docs
    except Exception as e:
        logger.error(f"检索相关文档时发生错误: {str(e)}", exc_info=True)
        raise

# 流式生成回答
def generate_answer_stream(qa_chain, question):
    """流式生成回答"""
    try:
        show_progress("正在调用模型生成回答（流式输出）...")
        
        # 使用流式调用，确保键名与prompt模板匹配
        for chunk in qa_chain.stream({"query": question}):
            if isinstance(chunk, dict) and "result" in chunk:
                yield chunk["result"]
    except Exception as e:
        logger.error(f"流式生成回答时发生错误: {str(e)}", exc_info=True)
        yield f"生成回答时发生错误: {str(e)}"

# 生成回答
def generate_answer(qa_chain, question):
    """生成回答"""
    try:
        show_progress("正在调用模型生成回答（可能需要几秒钟，请耐心等待...）")
        
        # 添加思考时间的视觉反馈
        start_time = time.time()
        
        # 使用正确的键名调用
        result = qa_chain.invoke({"query": question})
        
        end_time = time.time()
        show_progress(f"回答生成完成，用时: {round(end_time - start_time, 2)}秒")
        return result
    except Exception as e:
        logger.error(f"生成回答时发生错误: {str(e)}", exc_info=True)
        raise

# 本地RAG问答主函数
def local_rag_qa(question, use_streaming=False, enable_memory=True):
    """本地RAG问答主函数"""
    try:
        # 检查缓存
        if cache and cache.get(question):
            cached_result = cache.get(question)
            show_progress("使用缓存回答")
            return cached_result
        
        show_progress(f"开始处理问题: {question}")
        
        # 1. 初始化嵌入模型
        show_progress("[步骤1/6] 正在初始化嵌入模型...")
        embeddings = initialize_embeddings()
        
        # 2. 加载向量数据库
        show_progress("[步骤2/6] 正在加载向量数据库...")
        db = load_vector_db(embeddings)
        
        # 3. 初始化语言模型
        show_progress("[步骤3/6] 正在初始化语言模型...")
        llm = initialize_llm()
        
        # 4. 创建会话内存（如果启用）
        memory = None
        if enable_memory and CONVERSATION_CONFIG.get("memory_type"):
            show_progress("[步骤4/6] 正在创建会话内存...")
            memory = create_conversation_memory()
        else:
            show_progress("[步骤4/6] 跳过会话内存创建（未启用）")
        
        # 5. 创建问答链
        show_progress("[步骤5/6] 正在创建问答链...")
        qa_chain = create_qa_chain(db, llm, memory)
        
        # 6. 执行检索（可选，用于调试）
        show_progress("[步骤6/6] 正在检索相关文档...")
        retriever = db.as_retriever(
            search_type=RETRIEVAL_CONFIG["search_type"],
            search_kwargs={"k": RETRIEVAL_CONFIG["k"]}
        )
        relevant_docs = perform_retrieval(retriever, question)
        
        # 7. 生成回答
        # if use_streaming:
        #     show_progress("正在流式生成回答...")
        #     # 流式输出模式
        #     full_answer = ""
        #     print("\nAI 回答：")
        #     print("=" * 50)
            
        #     for chunk in generate_answer_stream(qa_chain, question):
        #         print(chunk, end="", flush=True)
        #         full_answer += chunk
            
        #     print("\n" + "=" * 50)
            
        #     # 构造结果对象
        #     result = {
        #         "result": full_answer,
        #         "source_documents": relevant_docs
        #     }
    # 在交互式问答的“打印回答”部分修改
# 在交互式问答的“打印回答”部分修改
        if use_streaming:
            full_answer = ""
            print("\n" + "=#+"*60)
            print("🎯 AI 回答（流式输出）：")
            print("-"*60)
            
            for chunk in generate_answer_stream(qa_chain, question):
                print(chunk, end="", flush=True)
                full_answer += chunk
            
            print("\n" + "="*60 + "\n")
            result = {"result": full_answer, "source_documents": relevant_docs}
        else:
            # 常规模式
            show_progress("正在生成回答...")
            result = generate_answer(qa_chain, question)
        
        # 缓存结果
        if cache:
            cache.set(question, result)
        
        show_progress("问题处理完成")
        return result
        
    except KeyboardInterrupt:
        error_msg = "操作被用户中断"
        show_progress(f"错误: {error_msg}")
        logger.warning(error_msg)
        return {
            "result": error_msg,
            "source_documents": []
        }
    except Exception as e:
        error_msg = f"处理问题时发生错误: {str(e)}"
        show_progress(f"错误: {error_msg}")
        logger.error(error_msg, exc_info=True)
        # 返回错误信息，便于调用者处理
        return {
            "result": error_msg,
            "source_documents": []
        }

# 交互式问答函数
def interactive_qa():
    """交互式问答函数"""
    print("\n" + "=" * 50)
    print("欢迎使用RAG问答系统")
    print("您可以输入问题进行问答，输入'exit'或'quit'退出")
    print("输入'clear'清空缓存")
    print("=" * 50 + "\n")
    
    # 启动时清空缓存以避免之前测试问题的影响
    if cache:
        cache.cache = {}
        print("缓存已在启动时清空，确保使用新问题")
    
    while True:
        try:
            question = input("\n请输入您的问题: ")
            
            # 处理命令
            if question.lower() in ["exit", "quit", "退出"]:
                print("谢谢使用，再见！")
                break
            elif question.lower() == "clear":
                if cache:
                    # 清空缓存
                    cache.cache = {}
                    print("缓存已清空")
                else:
                    print("缓存未启用")
                continue
            elif not question.strip():
                print("请输入有效的问题")
                continue
            
            # 处理问题
            print(f"\n正在处理问题: {question}")
            result = local_rag_qa(question, use_streaming=True, enable_memory=True)
            
            # 打印参考的文档片段
            if result.get("source_documents"):
                print("\n参考的文档片段：")
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get("source", "未知来源")
                    file_type = doc.metadata.get("file_type", "未知类型")
                    print(f"\n第 {i} 个片段 [来源: {os.path.basename(source)} ({file_type})]:")
                    # print(f"{doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)
                    print(f"{doc.page_content[:15]}..." if len(doc.page_content) > 15 else doc.page_content)
        except Exception as e:
            print(f"交互过程中发生错误: {str(e)}")
            logger.error(f"交互错误: {str(e)}", exc_info=True)

# 主函数：测试并打印结果
def main():
    try:
        import sys
        # 检查是否以交互式模式运行
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            # 交互式模式
            interactive_qa()
        elif len(sys.argv) > 1:
            # 从命令行获取问题
            question = " ".join(sys.argv[1:])
            show_progress(f"用户问题：{question}")
            
            # 执行问答
            show_progress("开始RAG流程处理，请稍候...")
            result = local_rag_qa(question, use_streaming=True)

            # 打印参考的文档片段
            if result.get("source_documents"):
                print("\n参考的文档片段：")
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get("source", "未知来源")
                    print(f"\n第 {i} 个片段 [来源: {source}]:")
                    print(f"{doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)
        else:
            # 默认模式，提示用户使用方法
            print("\n使用方法：")
            print("  1. 交互式问答: python script.py --interactive")
            print("  2. 直接提问: python script.py 你的问题")
    
    except KeyboardInterrupt:
        print("\n操作已中断")
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        show_progress(f"错误: {error_msg}")
        print(f"执行失败: {error_msg}")
        logger.error(error_msg, exc_info=True)

# 程序入口
if __name__ == "__main__":
    main()
        
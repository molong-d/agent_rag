import os
import sys
import time
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_environment():
    """测试环境配置和优化功能"""
    print("="*60)
    print("欢迎使用RAG项目优化功能测试脚本")
    print("此脚本将测试所有主要优化功能是否正常工作")
    print("="*60)
    
    all_tests_passed = True
    
    # 测试1: 检查配置文件和导入
    print("\n[测试1: 配置文件检查]")
    try:
        from config import PATHS, MODEL_CONFIG, RETRIEVAL_CONFIG, LOGGING_CONFIG, CACHE_CONFIG, CONVERSATION_CONFIG
        print("✓ 配置文件导入成功")
        print(f"  - 知识库路径: {PATHS['knowledge_docs']}")
        print(f"  - 向量数据库路径: {PATHS['chroma_data']}")
        print(f"  - 嵌入模型: {MODEL_CONFIG['embedding_model']}")
        print(f"  - LLM模型: {MODEL_CONFIG['ollama_model']}")
        print(f"  - 缓存配置: {'已启用' if CACHE_CONFIG['enabled'] else '未启用'}")
        print(f"  - 会话记忆: {'已启用' if CONVERSATION_CONFIG.get('memory_type') else '未启用'}")
    except Exception as e:
        print(f"✗ 配置文件测试失败: {str(e)}")
        logger.error(f"配置文件测试失败: {str(e)}", exc_info=True)
        all_tests_passed = False
    
    # 测试2: 检查目录结构
    print("\n[测试2: 目录结构检查]")
    required_dirs = [
        PATHS['knowledge_docs'],
        PATHS['chroma_data'],
        PATHS['model_path'],
        LOGGING_CONFIG['log_file']
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"⚠ 目录不存在: {dir_path} (将尝试创建)")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✓ 目录已创建: {dir_path}")
            except Exception as e:
                print(f"✗ 目录创建失败: {str(e)}")
                logger.error(f"目录创建失败: {str(e)}", exc_info=True)
    
    # 测试3: 检查文档格式支持
    print("\n[测试3: 文档格式支持检查]")
    try:
        # 检查导入是否成功
        import pypdf
        import docx
        import openpyxl
        import pptx
        import pandas
        print("✓ 文档格式支持库导入成功")
        print("  - PDF支持: ✓")
        print("  - Word支持: ✓")
        print("  - Excel支持: ✓")
        print("  - PowerPoint支持: ✓")
        print("  - CSV支持: ✓")
    except ImportError as e:
        print(f"⚠ 部分文档支持库可能未安装: {str(e)}")
        print("  请运行: pip install -r requirements.txt")
    
    # 测试4: 检查RAG初始化功能
    print("\n[测试4: RAG初始化功能检查]")
    try:
        import rag_init
        print("✓ RAG初始化模块导入成功")
        # 检查文档处理函数是否存在
        required_functions = ['load_documents', 'split_documents', 'init_vector_db']
        for func in required_functions:
            if hasattr(rag_init, func):
                print(f"  - {func}: ✓")
            else:
                print(f"  - {func}: ✗ 未找到")
                all_tests_passed = False
    except Exception as e:
        print(f"✗ RAG初始化功能测试失败: {str(e)}")
        logger.error(f"RAG初始化功能测试失败: {str(e)}", exc_info=True)
        all_tests_passed = False
    
    # 测试5: 检查代理功能
    print("\n[测试5: 代理功能检查]")
    try:
        from agent import SimpleCache, initialize_embeddings, load_vector_db, initialize_llm
        print("✓ 代理功能模块导入成功")
        print("  - 缓存系统: ✓")
        print("  - 嵌入模型初始化: ✓")
        print("  - 向量数据库加载: ✓")
        print("  - LLM初始化: ✓")
    except Exception as e:
        print(f"✗ 代理功能测试失败: {str(e)}")
        logger.error(f"代理功能测试失败: {str(e)}", exc_info=True)
        all_tests_passed = False
    
    # 测试6: 内存缓存功能测试
    if CACHE_CONFIG['enabled']:
        print("\n[测试6: 缓存功能测试]")
        try:
            from agent import SimpleCache
            cache = SimpleCache(max_size=10, ttl=10)
            
            # 添加测试数据
            cache.set("test_question", {"result": "test_answer"})
            
            # 检查缓存
            cached_result = cache.get("test_question")
            if cached_result and cached_result["result"] == "test_answer":
                print("✓ 缓存功能测试通过")
            else:
                print("✗ 缓存功能测试失败: 无法获取缓存数据")
                all_tests_passed = False
        except Exception as e:
            print(f"✗ 缓存功能测试失败: {str(e)}")
            logger.error(f"缓存功能测试失败: {str(e)}", exc_info=True)
            all_tests_passed = False
    
    # 生成测试报告
    print("\n" + "="*60)
    print("测试报告摘要")
    print("="*60)
    
    if all_tests_passed:
        print("🎉 所有测试通过！RAG项目优化功能正常工作。")
        print("\n接下来您可以：")
        print("1. 将文档放入知识库目录")
        print("2. 运行 'python rag_init.py' 初始化知识库")
        print("3. 运行 'python agent.py --interactive' 启动交互式问答")
    else:
        print("⚠ 部分测试失败，请查看日志文件进行排查。")
        print("\n建议操作：")
        print("1. 检查依赖是否安装完整: pip install -r requirements.txt")
        print("2. 确保Ollama服务正在运行")
        print("3. 验证配置文件中的路径设置")
    
    print("="*60)
    print(f"测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"详细日志已保存至: test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

if __name__ == "__main__":
    try:
        test_environment()
    except KeyboardInterrupt:
        print("\n\n测试已被用户中断")
    except Exception as e:
        print(f"\n\n测试过程中发生未预期的错误: {str(e)}")
        logger.error(f"测试过程中发生未预期的错误: {str(e)}", exc_info=True)
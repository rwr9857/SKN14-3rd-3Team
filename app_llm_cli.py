import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langchain_core.callbacks.base import BaseCallbackHandler
from rag_indexer_class import IndexConfig, RAGIndexer
from utils.index import image_to_base64

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def search_vector_db_image(img_path):
    """백터 디비에서 이미지의 모델을 가져온다"""

    # 설정 생성
    config = IndexConfig(
        persistent_directory="./chroma",
        collection_name="imgs",
        embedding_model="text-embedding-3-small",
    )

    # 인덱서 생성 및 실행
    indexer = RAGIndexer(config)

    # 이미지 로드해서 모델명 검색
    img_base64 = image_to_base64(img_path)

    # 유사도 검색
    model_nm = indexer.search_and_show(img_base64)
    return model_nm

class StreamingCallbackHandler(BaseCallbackHandler):
    """스트리밍 콜백 핸들러"""
    def __init__(self):
        self.tokens = []
        self.finished = False
    
    def on_llm_new_token(self, token: str) -> None:
        """새 토큰이 생성될 때 호출"""
        self.tokens.append(token)
    
    def on_llm_end(self) -> None:
        """LLM 응답 완료 시 호출"""
        self.finished = True

class SmartApplianceAssistant:
    def __init__(self):
        self.embeddings_model = "text-embedding-3-small"
        self.collection_name = "manuals"
        self.vector_db_dir = "./chroma"
        self.img_collection_name = "imgs"
        
        # LLM 초기화 (스트리밍 지원)
        self.llm = ChatOpenAI(
            model=MODEL_NAME, 
            temperature=0.3,
            streaming=True
        )
        
        # 분석용 LLM (스트리밍 불필요)
        self.analysis_llm = ChatOpenAI(
            model=MODEL_NAME, 
            temperature=0.1,
            streaming=False
        )
        
        # RAG 인덱서 초기화
        self._initialize_indexers()
        
        # 웹 검색 도구 초기화
        self.tavily_tool = TavilySearch(max_results=5)
        
        # 프롬프트 템플릿 초기화
        self._initialize_prompts()
        
        # 체인 구성
        self._setup_chains()
    
    def _initialize_indexers(self):
        """인덱서 초기화"""
        # 매뉴얼 텍스트용 인덱서
        self.text_config = IndexConfig(
            persistent_directory=self.vector_db_dir,
            collection_name=self.collection_name,
            embedding_model=self.embeddings_model,
        )
        self.text_indexer = RAGIndexer(self.text_config)
        
        # 이미지용 인덱서
        self.img_config = IndexConfig(
            persistent_directory=self.vector_db_dir,
            collection_name=self.img_collection_name,
            embedding_model=self.embeddings_model,
        )
        self.img_indexer = RAGIndexer(self.img_config)
        
        # 리트리버 설정
        self.retriever = self.text_indexer.vectordb.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 8, "fetch_k": 20}
        )
    
    def _initialize_prompts(self):
        """프롬프트 템플릿 초기화"""
        # 쿼리 분석 프롬프트
        self.query_analysis_prompt = ChatPromptTemplate.from_messages([
            ('system', """당신은 사용자의 질문을 분석하는 전문가입니다. 
            주어진 질문에서 다음을 추출하세요:
            
            1. 주요 키워드 (3-5개)
            2. 질문의 핵심 주제
            3. 구체적인 조건이나 요구사항
            4. 답변에서 다뤄야 할 세부 사항들
            
            JSON 형식으로 출력하세요:
            {{
                "keywords": ["키워드1", "키워드2", "키워드3"],
                "main_topic": "주제",
                "conditions": ["조건1"],
                "details": ["세부사항1"]
            }}"""),
            ("human", "질문: {query}"),
        ])
        
        # 메인 답변 프롬프트
        self.main_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 스마트한 가전 도우미입니다. Tree of Thoughts 방법론을 사용하여 체계적으로 답변하세요.
            
            ## 답변 지침
            1. 질문을 분석한 후 관련 정보를 수집하여 체계적으로 답변하세요
            2. 조건들을 나열하기보다는 통합하여 하나의 흐름으로 설명하세요
            3. 반복되거나 유사한 내용을 중복해서 설명하지 마세요
            4. 논리적 구조를 갖춘 명확한 문단 형태로 답변하세요
            5. 필요 시 예시나 유사 상황을 들어 이해를 도우세요
            
            ## 출력 형식
            ### 해결책
            - [체계적인 통합 설명을 한 문단 이상으로 기술]
            
            ### 추가 안내
            - [관련된 팁이나 참고 정보가 있으면 제공]
            """),
            ("human", """
            질문: {query}
            분석 결과: {analysis}
            
            관련 정보:
            {context}
            """),
        ])
    
    def _setup_chains(self):
        """체인 구성"""
        # 쿼리 분석 체인 (스트리밍 불필요)
        self.query_analysis_chain = self.query_analysis_prompt | self.analysis_llm | StrOutputParser()
        
        # 메인 답변 체인 (스트리밍 지원)
        self.answer_chain = self.main_answer_prompt | self.llm
    
    def search_vector_db_image(self, img_path: str) -> str:
        """벡터 DB에서 이미지의 모델을 가져온다"""
        try:
            img_base64 = image_to_base64(img_path)
            model_nm = self.img_indexer.search_and_show(img_base64)
            return model_nm if model_nm != -1 else "확인불가"
        except Exception as e:
            logger.error(f"이미지 검색 오류: {e}")
            return "확인불가"
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF 텍스트 추출"""
        try:
            return extract_text(pdf_path)
        except Exception as e:
            logger.error(f"PDF 읽기 실패 {pdf_path}: {e}")
            return ""
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """질문 분석"""
        try:
            analysis_result = self.query_analysis_chain.invoke({"query": query})
            return json.loads(analysis_result)
        except Exception as e:
            logger.error(f"질문 분석 오류: {e}")
            return {
                "keywords": [query],
                "main_topic": query,
                "conditions": [],
                "details": []
            }
    
    def search_web(self, query: str) -> List[Document]:
        """웹 검색"""
        web_docs = []
        try:
            search_result = self.tavily_tool.invoke({"query": query})
            web_results = search_result.get("results", [])
            
            for item in web_results:
                content = item.get("content", "")
                url = item.get("url", "")
                
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": url, 
                            "title": item.get("title", ""),
                            "type": "web"
                        }
                    )
                    web_docs.append(doc)
        except Exception as e:
            logger.error(f"웹 검색 오류: {e}")
        
        return web_docs
    
    def search_vector_db(self, keywords: List[str]) -> List[Document]:
        """벡터 DB 검색"""
        vector_docs = []
        for keyword in keywords:
            try:
                docs = self.retriever.invoke(keyword)
                for doc in docs:
                    doc.metadata["type"] = "vector"
                vector_docs.extend(docs)
            except Exception as e:
                logger.error(f"벡터 검색 오류 ({keyword}): {e}")
        
        return vector_docs
    
    def retrieve_contexts(self, query: str, analysis: Dict[str, Any]) -> List[Document]:
        """모든 소스에서 컨텍스트 검색"""
        all_contexts = []
        
        # 웹 검색
        web_docs = self.search_web(query)
        all_contexts.extend(web_docs)
        
        # 벡터 DB 검색
        keywords = analysis.get("keywords", [query])
        vector_docs = self.search_vector_db(keywords)
        all_contexts.extend(vector_docs)
        
        # 중복 제거 및 정렬
        unique_contexts = self._deduplicate_contexts(all_contexts)
        
        return unique_contexts
    
    def _deduplicate_contexts(self, contexts: List[Document]) -> List[Document]:
        """컨텍스트 중복 제거"""
        seen_content = set()
        unique_contexts = []
        
        for doc in contexts:
            content_hash = hash(doc.page_content[:100])  # 첫 100자로 해시
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_contexts.append(doc)
        
        return unique_contexts
    
    def format_context(self, contexts: List[Document]) -> str:
        """컨텍스트 포맷팅"""
        if not contexts:
            return "관련 정보를 찾을 수 없습니다."
        
        formatted_contexts = []
        for _, doc in enumerate(contexts[:10]):  # 최대 10개만 사용
            source = doc.metadata.get("source", "알 수 없음")
            doc_type = doc.metadata.get("type", "unknown")
            content = doc.page_content[:500]  # 최대 500자
            
            formatted_contexts.append(f"[{doc_type.upper()}] {content}\n출처: {source}")
        
        return "\n\n".join(formatted_contexts)
    
    def process_query_with_image(self, query: str, image_path: str) -> str:
        """이미지가 포함된 질문 처리"""
        model_code = self.search_vector_db_image(image_path)
        return f"{query} (모델코드: {model_code})"
    
    def chat_stream(self, query: str, image_path: Optional[str] = None, history: List[Dict[str, str]] = None) -> Generator[str, None, None]:
        """스트리밍 채팅 함수"""
        if history is None:
            history = []
        
        try:
            # 이미지가 있으면 모델 코드 추가
            if image_path:
                query = self.process_query_with_image(query, image_path)
            
            # 1. 질문 분석 (스트리밍 없이)
            yield "🔍 질문을 분석하고 있습니다...\n\n"
            analysis = self.analyze_query(query)
            logger.info(f"질문 분석 완료: {analysis}")
            
            # 2. 컨텍스트 검색
            yield "📚 관련 정보를 검색하고 있습니다...\n\n"
            contexts = self.retrieve_contexts(query, analysis)
            formatted_context = self.format_context(contexts)
            
            # 3. 답변 생성 (스트리밍)
            yield "💡 답변을 생성하고 있습니다...\n\n"
            
            # 스트리밍 콜백 핸들러 설정
            callback_handler = StreamingCallbackHandler()
            
            # 스트리밍으로 답변 생성
            for chunk in self.answer_chain.stream(
                {
                    "query": query,
                    "analysis": json.dumps(analysis, ensure_ascii=False, indent=2),
                    "context": formatted_context
                },
                config={"callbacks": [callback_handler]}
            ):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                
        except Exception as e:
            logger.error(f"스트리밍 챗봇 처리 오류: {e}")
            yield f"❌ 죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
    
    async def chat_stream_async(self, query: str, image_path: Optional[str] = None, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """비동기 스트리밍 채팅 함수"""
        if history is None:
            history = []
        
        try:
            # 이미지가 있으면 모델 코드 추가
            if image_path:
                query = self.process_query_with_image(query, image_path)
            
            # 1. 질문 분석 (비동기)
            yield "🔍 질문을 분석하고 있습니다...\n\n"
            analysis = await self.query_analysis_chain.ainvoke({"query": query})
            analysis_data = json.loads(analysis)
            logger.info(f"질문 분석 완료: {analysis_data}")
            
            # 2. 컨텍스트 검색
            yield "📚 관련 정보를 검색하고 있습니다...\n\n"
            contexts = self.retrieve_contexts(query, analysis_data)
            formatted_context = self.format_context(contexts)
            
            # 3. 답변 생성 (비동기 스트리밍)
            yield "💡 답변을 생성하고 있습니다...\n\n"
            
            async for chunk in self.answer_chain.astream({
                "query": query,
                "analysis": json.dumps(analysis_data, ensure_ascii=False, indent=2),
                "context": formatted_context
            }):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                
        except Exception as e:
            logger.error(f"비동기 스트리밍 챗봇 처리 오류: {e}")
            yield f"❌ 죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
    
    def chat(self, query: str, image_path: Optional[str] = None, history: List[Dict[str, str]] = None) -> str:
        """일반 채팅 함수 (스트리밍 없음)"""
        response_parts = []
        for chunk in self.chat_stream(query, image_path, history):
            response_parts.append(chunk)
        return "".join(response_parts)

# 편의 함수들
def run_chatbot_stream(query: str, image_path: Optional[str] = None, history: List[Dict[str, str]] = None) -> Generator[str, None, None]:
    """스트리밍 챗봇 실행 함수"""
    assistant = SmartApplianceAssistant()
    yield from assistant.chat_stream(query, image_path, history)

async def run_chatbot_stream_async(query: str, image_path: Optional[str] = None, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
    """비동기 스트리밍 챗봇 실행 함수"""
    assistant = SmartApplianceAssistant()
    async for chunk in assistant.chat_stream_async(query, image_path, history):
        yield chunk

def run_chatbot(query: str, image_path: Optional[str] = None, history: List[Dict[str, str]] = None) -> str:
    """기존 함수와의 호환성을 위한 래퍼 함수"""
    assistant = SmartApplianceAssistant()
    return assistant.chat(query, image_path, history)

# 사용 예시
if __name__ == "__main__":
    import time
    
    # 동기 스트리밍 예시
    def sync_streaming_example():
        print("=== 동기 스트리밍 예시 ===")
        assistant = SmartApplianceAssistant()
        
        query = "에어컨 필터 청소 방법을 알려주세요"
        print(f"질문: {query}\n")
        
        for chunk in assistant.chat_stream(query):
            print(chunk, end="", flush=True)
            time.sleep(0.01)  # 실제 스트리밍 효과를 위한 지연
        print("\n")
    
    # 비동기 스트리밍 예시
    async def async_streaming_example():
        print("\n=== 비동기 스트리밍 예시 ===")
        assistant = SmartApplianceAssistant()
        
        query = "냉장고에서 이상한 소리가 나는데 어떻게 해야 하나요?"
        print(f"질문: {query}\n")
        
        async for chunk in assistant.chat_stream_async(query):
            print(chunk, end="", flush=True)
            await asyncio.sleep(0.01)  # 실제 스트리밍 효과를 위한 지연
        print("\n")
    
    # 일반 채팅 예시
    def normal_chat_example():
        print("\n=== 일반 채팅 예시 ===")
        assistant = SmartApplianceAssistant()
        
        query = "세탁기 배수가 안 되는 문제 해결법"
        print(f"질문: {query}\n")
        
        response = assistant.chat(query)
        print(f"답변: {response}\n")
    
    # 예시 실행
    sync_streaming_example()
    asyncio.run(async_streaming_example())
    normal_chat_example()
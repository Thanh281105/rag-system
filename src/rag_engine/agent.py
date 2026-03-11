import os
from collections import deque
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from src.rag_engine.retriever import LegalRetriever


class LegalAgent:
    def __init__(self, db_dir, groq_api_key, memory_k: int = 5):
        # 1. Khởi tạo LLM
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            api_key=groq_api_key,
        )
        self.retriever = LegalRetriever(db_dir)

        # 2. Conversation Memory — deque, không dùng langchain.memory (deprecated)
        self._memory: deque[dict] = deque(maxlen=memory_k)

        # 3. PROMPT CHÍNH
        self.prompt = PromptTemplate(
            template="""Bạn là một luật sư AI chuyên nghiệp tại Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA VÀO ĐÚNG ngữ cảnh pháp lý được cung cấp dưới đây.

=== QUY TẮC BẮT BUỘC ===

1. ƯU TIÊN KIỂM TRA NGÀNH CẤM & NGÀNH CÓ ĐIỀU KIỆN:
   Nếu câu hỏi liên quan đến việc thành lập, kinh doanh, hay hoạt động trong một ngành nghề cụ thể,
   BƯỚC ĐẦU TIÊN là kiểm tra xem ngành đó có xuất hiện trong:
   - Điều 6 Luật Đầu tư (Ngành, nghề CẤM đầu tư kinh doanh)
   - Điều 7 + Phụ lục IV Luật Đầu tư (Ngành, nghề có ĐIỀU KIỆN)
   Nếu ngành bị cấm → trả lời ngay là KHÔNG ĐƯỢC PHÉP, không cần phân tích thêm.

2. NGUYÊN TẮC SUY LUẬN CÓ ĐIỀU KIỆN:
   Chỉ áp dụng nguyên tắc "không cấm = được phép" khi:
   a) Ngữ cảnh đã có đủ thông tin về lĩnh vực đó, VÀ
   b) Đã kiểm tra xác nhận ngành đó KHÔNG có trong danh sách cấm/có điều kiện.
   TUYỆT ĐỐI không suy luận khi thiếu dữ liệu.

3. KHI KHÔNG TÌM THẤY THÔNG TIN:
   Không được tự suy luận hay đưa ra kết luận. Phải trả lời:
   "Tôi chưa tìm thấy quy định cụ thể trong cơ sở dữ liệu hiện tại.
   Bạn nên kiểm tra thêm: Luật Đầu tư 2020 (Điều 6, 7), các Nghị định
   hướng dẫn và văn bản pháp luật chuyên ngành liên quan."

4. SO SÁNH LOẠI HÌNH DOANH NGHIỆP:
   Nếu câu hỏi nhắc đến loại hình cụ thể (Công ty Cổ phần, TNHH...),
   chỉ lấy quy định đúng chương/loại hình đó, không áp dụng chéo.

5. KHÔNG BỊA ĐẶT: Mọi thông tin phải có nguồn từ ngữ cảnh được cung cấp.

6. TRÍCH DẪN NGUỒN BẮT BUỘC: Ghi rõ (Nguồn: [Tên Luật] | [Chương] | [Điều] | [Khoản/Điểm]).

7. CẢNH BÁO RỦI RO PHÁP LÝ: Cuối mỗi câu trả lời về ngành nghề kinh doanh,
   nhắc người dùng xác nhận lại với cơ quan nhà nước có thẩm quyền vì pháp luật
   có thể thay đổi theo thời gian.

=== LỊCH SỬ HỘI THOẠI ===
{chat_history}

=== NGỮ CẢNH PHÁP LÝ TÌM ĐƯỢC ===
{context}

=== CÂU HỎI CỦA NGƯỜI DÙNG ===
{question}

=== CÂU TRẢ LỜI CỦA LUẬT SƯ AI ===
""",
            input_variables=["chat_history", "context", "question"],
        )

        # 4. PROMPT MULTI-QUERY: sinh 3 biến thể query để tăng recall
        self.multi_query_prompt = PromptTemplate(
            template="""Bạn là chuyên gia tra cứu pháp luật Việt Nam.
Nhiệm vụ: Từ câu hỏi gốc, hãy sinh ra ĐÚNG 3 cách diễn đạt khác nhau dưới dạng thuật ngữ pháp lý
để tối ưu tìm kiếm trong cơ sở dữ liệu luật.

QUY TẮC:
- Query 1: Dịch thẳng sang thuật ngữ pháp lý chính xác nhất.
- Query 2: Mở rộng thêm ngữ cảnh điều luật liên quan (Điều mấy, Luật nào).
- Query 3: Diễn đạt theo góc độ ngược lại hoặc hệ quả pháp lý.
- NẾU câu hỏi về mở công ty / kinh doanh ngành nghề: BẮT BUỘC 1 trong 3 query phải chứa
  "ngành nghề cấm đầu tư kinh doanh Điều 6 Luật Đầu tư".
- NẾU câu hỏi về chuyên ngành khác: KHÔNG thêm từ khóa Luật Đầu tư.

Trả về ĐÚNG 3 dòng, mỗi dòng là 1 query, KHÔNG đánh số, KHÔNG giải thích.

Lịch sử hội thoại:
{chat_history}

Câu hỏi gốc: {question}

3 query pháp lý:""",
            input_variables=["chat_history", "question"],
        )

    # ------------------------------------------------------------------ #
    #  Memory                                                              #
    # ------------------------------------------------------------------ #

    def _get_chat_history(self) -> str:
        if not self._memory:
            return "(Chưa có lịch sử hội thoại)"
        lines = []
        for turn in self._memory:
            lines.append(f"Người dùng: {turn['input']}")
            lines.append(f"Luật sư AI: {turn['output']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Main                                                                #
    # ------------------------------------------------------------------ #

    def ask(self, user_question: str) -> str:
        chat_history = self._get_chat_history()

        # BƯỚC 1: Multi-Query Generation
        # LLM sinh 3 biến thể query để tăng recall khi search
        mq_chain = self.multi_query_prompt | self.llm
        raw_queries = mq_chain.invoke(
            {"question": user_question, "chat_history": chat_history}
        ).content

        # Parse 3 dòng → list, luôn đặt query gốc ở đầu (dùng để rerank)
        expanded_queries = [q.strip() for q in raw_queries.strip().splitlines() if q.strip()]
        all_queries = [user_question] + expanded_queries[:3]  # tối đa 4 queries

        print(f"\n🧠 Multi-Query ({len(all_queries)} biến thể):")
        for i, q in enumerate(all_queries):
            label = "gốc " if i == 0 else f"  {i}  "
            print(f"   [{label}] {q}")

        # BƯỚC 2: Multi-Query Parallel Hybrid Search + Rerank + Threshold
        relevant_docs = self.retriever.get_relevant_laws(all_queries, k=15)

        if not relevant_docs:
            return (
                "Xin lỗi, hệ thống không tìm thấy tài liệu pháp lý nào liên quan.\n"
                "⚠️ Bạn nên tham khảo trực tiếp Luật Đầu tư 2020 và các văn bản "
                "pháp luật chuyên ngành tại cổng thông tin pháp luật chính thức."
            )

        print(f"\n--- 🕵️ DEBUG: {len(relevant_docs)} KHOẢN LUẬT SAU KHI LỌC ---")
        for i, doc in enumerate(relevant_docs):
            meta = doc.metadata
            print(
                f"Top {i+1}: {meta.get('loai_van_ban')} | {meta.get('chuong')} | "
                f"{meta.get('dieu')} | {meta.get('khoan')}"
            )
        print("----------------------------------------------------------\n")

        # BƯỚC 3: Context Truncation — giới hạn độ dài, tránh vượt context window
        context_text = LegalRetriever.truncate_context(relevant_docs)

        # BƯỚC 4: Sinh câu trả lời
        chain = self.prompt | self.llm
        response = chain.invoke({
            "chat_history": chat_history,
            "context": context_text,
            "question": user_question,
        })
        answer = response.content

        # BƯỚC 5: Lưu memory
        self._memory.append({"input": user_question, "output": answer})

        return answer

    def reset_memory(self):
        self._memory.clear()
        print("🔄 Đã xóa lịch sử hội thoại.")

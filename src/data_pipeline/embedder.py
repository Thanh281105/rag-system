import json
import os
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ============================================================
#  MODEL CONFIG — đổi model ở đây, không cần sửa logic bên dưới
# ============================================================
BI_ENCODER_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
#   Lý do chọn: Train trên corpus pháp lý + báo chí Tiếng Việt,
#   hiểu được ngữ cảnh câu chữ chuyên ngành tốt hơn nhiều so với
#   vietnamese-sbert (vốn chỉ train trên dữ liệu đa dụng).
#
#   ⚠️  LƯU Ý QUAN TRỌNG: Nếu bạn đổi BI_ENCODER_MODEL, BẮT BUỘC
#   phải chạy clear_and_rebuild() để tạo lại toàn bộ Vector DB.
#   Vector cũ (từ model cũ) và vector mới KHÔNG tương thích nhau.
# ============================================================


class LegalEmbedder:
    def __init__(self, processed_dir, persist_directory):
        self.processed_dir = processed_dir
        self.persist_directory = persist_directory

        print(f"1. Đang khởi động Bi-Encoder: [{BI_ENCODER_MODEL}]...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=BI_ENCODER_MODEL,
            model_kwargs={"device": "cpu"},       # đổi "cuda" nếu có GPU
            encode_kwargs={"normalize_embeddings": True},  # chuẩn hóa vector → cosine similarity chính xác hơn
        )

        print("2. Đang kết nối tới Vector DB (Chroma)...")
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="he_thong_phap_luat",
        )

    def _load_documents(self):
        documents = []
        doc_ids = []
        seen_ids = set()

        json_files = glob.glob(os.path.join(self.processed_dir, "*.json"))
        print(f"3. Tìm thấy {len(json_files)} bộ luật. Đang gom dữ liệu...")

        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    ten_luat = item["metadata"]["loai_van_ban"]
                    chuong   = item["metadata"]["chuong"]
                    dieu     = item["metadata"]["dieu"]
                    khoan    = item["metadata"].get("khoan", "Khong_co_khoan")

                    unique_id = f"{ten_luat}_{chuong}_{dieu}_{khoan}"

                    # Chống trùng lặp ID
                    original_id = unique_id
                    counter = 1
                    while unique_id in seen_ids:
                        unique_id = f"{original_id}_part_{counter}"
                        counter += 1
                    seen_ids.add(unique_id)

                    doc = Document(
                        page_content=item["noi_dung"],
                        metadata=item["metadata"],
                    )
                    documents.append(doc)
                    doc_ids.append(unique_id)

        return documents, doc_ids

    def create_vector_db(self):
        """Upsert các văn bản mới vào DB, không xóa dữ liệu cũ."""
        documents, doc_ids = self._load_documents()
        print(f"4. Đang Upsert {len(documents)} Khoản vào ChromaDB (batch 500)...")

        batch_size = 500
        total = len(documents)
        for i in range(0, total, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids  = doc_ids[i : i + batch_size]
            self.vector_db.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"   -> Đã nạp {min(i + batch_size, total)}/{total} Khoản...")

        print("🎉 THÀNH CÔNG! Đã cập nhật Vector DB (Upsert, không mất dữ liệu cũ).")

    def clear_and_rebuild(self):
        """
        Xóa toàn bộ DB và xây lại từ đầu.
        BẮT BUỘC gọi hàm này khi đổi BI_ENCODER_MODEL.
        """
        print("⚠️  Đang xóa toàn bộ Vector DB và xây lại...")
        self.vector_db.delete_collection()
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="he_thong_phap_luat",
        )
        self.create_vector_db()


if __name__ == "__main__":
    processed_dir = "../../data/processed"
    db_dir        = "../../data/vector_db"
    os.makedirs(db_dir, exist_ok=True)

    embedder = LegalEmbedder(processed_dir, db_dir)

    # Lần đầu chạy hoặc vừa đổi model → dùng clear_and_rebuild()
    # Chỉ thêm luật mới → dùng create_vector_db()
    embedder.clear_and_rebuild()
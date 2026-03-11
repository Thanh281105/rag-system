import re
import json
import os
import glob
from docx import Document

class LegalParser:
    def __init__(self, file_path, doc_name, nhom_van_ban):
        self.file_path = file_path
        self.doc_name = doc_name            # Tên văn bản (VD: Luat Doanh Nghiep, Nghi Dinh 01 2021)
        self.nhom_van_ban = nhom_van_ban    # Phân loại: "Luat" hoặc "Nghi_dinh"
        self.parsed_data = []
        
        # Gọi Đầu đọc vạn năng ngay khi khởi tạo
        self.lines = self._read_file()

    def _read_file(self):
        """Đầu đọc vạn năng: Tự nhận diện đuôi file để dùng đúng engine đọc"""
        ext = os.path.splitext(self.file_path)[1].lower()
        lines = []
        try:
            if ext == '.docx':
                doc = Document(self.file_path)
                lines = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            elif ext == '.txt':
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"❌ Lỗi đọc file {self.file_path}: {e}")
        return lines

    def parse_structure(self):
        if not self.lines:
            return

        current_chuong = "Khong_co_chuong"
        current_dieu = "Khong_co_dieu"
        ten_dieu = ""
        current_khoan = "Phan_chung"
        current_content = []

        print(f"Đang bóc tách [{self.nhom_van_ban}]: {self.doc_name}...")

        for text in self.lines:
            # 1. Nhận diện Chương
            if re.match(r"^CHƯƠNG\s+[IVXLCDM]+", text, re.IGNORECASE):
                if current_content:
                    self._save_chunk(current_chuong, current_dieu, ten_dieu, current_khoan, current_content)
                    current_content = []
                current_chuong = text
                continue

            # 2. Nhận diện Điều (Thêm [a-zA-Z]* để bắt các điều sửa đổi như Điều 1a, Điều 1b)
            dieu_match = re.match(r"^(Điều\s+\d+[a-zA-Z]*)\.(.*)", text, re.IGNORECASE)
            if dieu_match:
                if current_content:
                    self._save_chunk(current_chuong, current_dieu, ten_dieu, current_khoan, current_content)

                current_dieu = dieu_match.group(1).strip()
                ten_dieu = dieu_match.group(2).strip()
                current_khoan = "Phan_chung"
                current_content = []
                continue

            # 3. Nhận diện Khoản
            khoan_match = re.match(r"^(\d+)\.\s+(.*)", text)
            if khoan_match and int(khoan_match.group(1)) <= 20 and current_dieu != "Khong_co_dieu":
                if current_content:
                    self._save_chunk(current_chuong, current_dieu, ten_dieu, current_khoan, current_content)

                current_khoan = f"Khoản {khoan_match.group(1)}"
                current_content = [text]
                continue

            # 4. Nhận diện Điểm (a, b, c...)
            diem_match = re.match(r"^([a-z])\)\s+(.*)", text)
            if diem_match and current_dieu != "Khong_co_dieu":
                current_content.append(text)
                continue

            if current_dieu != "Khong_co_dieu":
                current_content.append(text)

        # Lưu chunk cuối cùng
        if current_content:
            self._save_chunk(current_chuong, current_dieu, ten_dieu, current_khoan, current_content)

        print(f"-> Hoàn thành! Cắt nhỏ được {len(self.parsed_data)} đoạn (Khoản).")

    def _save_chunk(self, chuong, dieu, ten_dieu, khoan, content_list):
        if not content_list:
            return

        context_header = f"{dieu}. {ten_dieu}"
        body_text = "\n".join(content_list)
        noi_dung_final = f"{context_header}\n{body_text}"

        chunk = {
            "metadata": {
                "nhom_van_ban": self.nhom_van_ban,  # "Luat" hoặc "Nghi_dinh"
                "loai_van_ban": self.doc_name,      # "Luat Doanh Nghiep" (để embedder tạo ID không bị lỗi)
                "chuong": chuong,
                "dieu": dieu,
                "ten_dieu": ten_dieu,
                "khoan": khoan,
            },
            "noi_dung": noi_dung_final,
        }
        self.parsed_data.append(chunk)

    def export_to_json(self, output_path):
        if not self.parsed_data:
            return
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.parsed_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # KHAI BÁO CÁC KHO DỮ LIỆU
    DATA_SOURCES = {
        "Luat": "../../data/raw",
        "Nghi_dinh": "../../data/raw_nghidinh"
    }
    
    processed_dir = "../../data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    total_chunks = 0

    for nhom_van_ban, folder_path in DATA_SOURCES.items():
        if not os.path.exists(folder_path):
            print(f"⚠️ Không tìm thấy thư mục {folder_path}, bỏ qua...")
            continue
            
        print(f"\n📂 ĐANG XỬ LÝ KHO DỮ LIỆU: {nhom_van_ban.upper()}...")
        
        # Quét lấy cả file .docx và .txt
        all_files = glob.glob(os.path.join(folder_path, "*.*"))
        target_files = [f for f in all_files if f.endswith('.docx') or f.endswith('.txt')]
        
        if not target_files:
            print(f"   Trống! Không có file hợp lệ trong {folder_path}")
            
        for file_path in target_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            doc_name = base_name.replace("_", " ").title()
            output_file = os.path.join(processed_dir, f"{base_name}_parsed.json")

            parser = LegalParser(file_path, doc_name, nhom_van_ban)
            parser.parse_structure()
            parser.export_to_json(output_file)
            
            total_chunks += len(parser.parsed_data)
            print(f"   ✅ Đã xuất JSON: {output_file}")

    print(f"\n🎉 HOÀN TẤT! Tổng cộng {total_chunks} Khoản/Điều đã sẵn sàng để nạp vào Vector DB.")
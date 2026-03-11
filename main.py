import os
from dotenv import load_dotenv
from src.rag_engine.agent import LegalAgent


def main():
    # 1. Tải API Key từ file .env
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        print("❌ LỖI: Chưa tìm thấy GROQ_API_KEY trong file .env!")
        print("Vui lòng tạo file .env ở thư mục gốc và thêm dòng: GROQ_API_KEY=gsk_...")
        return

    print("=" * 60)
    print(" ⚖️  HỆ THỐNG TƯ VẤN PHÁP LUẬT AI (PHIÊN BẢN NÂNG CẤP)")
    print("=" * 60)

    # 2. Khởi tạo Agent
    db_dir = "data/vector_db"

    try:
        agent = LegalAgent(db_dir, groq_api_key)
    except Exception as e:
        print(f"❌ Lỗi khi khởi động AI: {e}")
        return

    print("\n✅ Hệ thống đã sẵn sàng!")
    print("   Gõ 'thoat' để kết thúc.")
    print("   Gõ 'reset' để xóa lịch sử hội thoại và bắt đầu lại.\n")

    # 3. Vòng lặp Chat
    while True:
        try:
            user_query = input("🧑 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt! Hẹn gặp lại.")
            break

        if not user_query:
            continue

        if user_query.lower() in ["thoat", "quit", "exit", "q"]:
            print("👋 Tạm biệt! Hẹn gặp lại.")
            break

        if user_query.lower() == "reset":
            agent.reset_memory()
            continue

        try:
            answer = agent.ask(user_query)
            print(f"\n🤖 Luật sư AI:\n{answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"\n❌ Có lỗi xảy ra trong quá trình xử lý: {e}\n")


if __name__ == "__main__":
    main() 
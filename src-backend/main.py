import requests

# === 1. Nhập input từ người dùng ===
gender = int(input("Nhập giới tính (1: Nam, 2: Nữ): "))
age = int(input("Nhập độ tuổi (0–6): "))

# === 2. Payload chuẩn bị gửi ===
payload = {
    "gender": gender,
    "age": age
}

# === 3. Gọi API video ===
video_url = "http://127.0.0.1:5001/recommend/video"
article_url = "http://127.0.0.1:5002/recommend/article"

try:
    res_video = requests.post(video_url, json=payload)
    res_article = requests.post(article_url, json=payload)

    video_data = res_video.json()
    article_data = res_article.json()

    # === 4. In kết quả ra màn hình ===
    print("\n🎬 Kết quả gợi ý VIDEO:")
    for item in video_data['recommendations']:
        print(f"Video ID: {item['item_id']} – Score: {item['score']}")

    print("\n📰 Kết quả gợi ý BÀI VIẾT:")
    for item in article_data['recommendations']:
        print(f"Article ID: {item['item_id']} – Score: {item['score']}")

except Exception as e:
    print("❌ Lỗi khi gọi backend:", e)

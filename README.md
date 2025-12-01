# Machine Learning Assignment – HK251

## Thông tin môn học
* Môn học: Học máy (Machine Learning)
* Mã môn học: CO3117
* Học kỳ: 251
* Giảng viên: TS.Lê Thành Sách

## Thông tin nhóm MNTV
| MSSV | Họ và tên | Email |
|------|-----------|-------|
| 2312420 | Huỳnh Đức Nhân | nhan.huynhgl272@hcmut.edu.vn |
| 2312097 | Nguyễn Thiện Minh | thienminha5k31@gmail.com |
| 2313638 | Nguyễn Lưu Khánh Trình | trinh.nguyenktmtbk0711@hcmut.edu.vn |
| 2313912 | Lê Công Vinh | vinh.le020705@hcmut.edu.vn |


🌐 **Landing Page:** [Comming soon](https://huynhnhan27.github.io/BTL_ML_251/)  
🔗 **Github Repo:** [Github](https://github.com/HuynhNhan27/BTL_ML_251)

---

## 🚀 Projects Overview

Repo này chứa 4 bài tập lớn của môn học *Học Máy – CO3117*:  

| Project | Domain | Status | Dataset | Colab |
|---------|--------|--------|------------------|-------|
| **BTL1 – Amazon Product** | Tabular Data | ✅ Completed | [Amazon Products](https://www.kaggle.com/datasets/ikramshah512/amazon-products-sales-dataset-42k-items-2025) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Y6CAMgL1Y0mev4UZJOi-FPIP7UMF4xSv?usp=sharing) |
| **BTL2 – Text Processing** | Text Data | ✅ Completed | [Emotion Detect](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text/data) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FpZT_pxSkoPX01GMBuddQoQ_bh7SBXdC?usp=sharing) |
| **BTL3 – Image Recognition** | Computer Vision | ✅ Completed | [Mineral Photos](https://www.kaggle.com/datasets/floriangeillon/mineral-photos) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JlpRaJXwa1ZzH9b3-W4kKNYHwkiTW2pD?usp=sharing) |
| **Extension – Advanced Topics** | Bayesian Network | ✅ Completed | [Student Depression](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1H6O0cE0q_EYJlBak-Nht_oQqTRyRVR_b?usp=sharing) |

---

## 📊 Mục tiêu bài tập lớn

Nội dung bài tập trải dài trên ba dạng dữ liệu phổ biến: dữ liệu dạng bảng, dữ liệu văn bản và dữ liệu ảnh. Mỗi dạng dữ liệu đều có những đặc thù riêng, yêu cầu những phương pháp xử lý và mô hình phù hợp. Các bước xử lý chính của cả ba dạng dữ liệu bao gồm:

- Phân tích dữ liệu khám phá (EDA)  
- Xử lý dữ liệu & Feature Engineering  
- Xây dựng pipeline Học máy truyền thống và/hoặc Học sâu  
- Huấn luyện, đánh giá & so sánh mô hình

Thông qua các bước xử lý đó, sinh viên hiểu được quy trình xử lý pipeline học máy truyền thống, rèn luyện được kỹ năng sử dụng các thư viện quan trọng như numpy, pandas, sklearn, .... Ngoài ra sinh viên còn rèn luyện được tinh thần hợp tác làm việc nhóm, tổ chức báo cáo khoa học.

---

## 📂 Repo Structure

```
ML251/
│── data/                       
    │── tên_data
        │── raw_data/           # Data gốc
        │── features/           # Features trích xuất từ data
        │── model_result/       # Kết quả của các mô hình
│── modules/                    # Các modules, utils tự viết được tái sử dụng
│── notebooks/                  # Notebooks cho các bài tập lớn
│── docs/                       # Github Page
|── report/                     # Report phần mở rộng và biên bản họp
│── README.md
│── requirements.txt
```

---

## ▶️ Usage

Clone repo và cài đặt dependencies:

```bash
git clone https://github.com/HuynhNhan27/BTL_ML_251.git
cd BTL_ML_251

# (Tuỳ chọn) tạo môi trường ảo
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

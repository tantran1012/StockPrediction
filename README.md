# Stock Prediction

# Machine Learning Project

## ℹ️ Thông tin nhóm

-   18120433 - Nguyễn Văn Lâm
-   18120644 - Nguyễn Cát Tường
-   1712149 - Trần Minh Tân

## 📃 Yêu cầu

-   Python > 3.10
-   VSCode hoặc Anaconda

## 📦 Hướng dẫn cài đặt

### Install các thư viện

#### 🐳 VSCode

-   Cài đặt thư viện

```sh
  pip install keras tensorflow xgboost alpha_vantage dash pandas matplotlib
```

-   Cài các Extensions (VSCode tự động yêu cầu khi mở project)
    -   Python
    -   Pylance
    -   Jupyter
    -   Jupyter Cell Tags
    -   Jupyter Keymap
    -   Jupyter Slide Show

#### 🐍 Anaconda

```sh
  conda install -c conda-force keras tensorflow xgboost alpha_vantage dash pandas matplotlib
```

### Chạy StockApplication

-   Mở terminal tại thư mục gốc và gõ lệnh bên dưới:

```sh
  cd APP
  python StockApplication.py
```

-   Hoặc vào thư mục APP, chạy nội dung trong file Run.ipynb

## Demo

🎬 Youtube:

## Các tính năng

-   Người dùng chọn một trong các phương pháp dự đoán
    -   XGBoost
    -   RNN
    -   LSTM
-   Người dùng chọn một hay nhiều đặc trưng để dự đoán
    -   Close, Price of Change
    -   RSI, Bolling Bands, Moving Average
-   Hiển thị giá dự đoán của timeframe kế tiếp
-   Lấy dữ liệu từ Alpha Vantage
    -   Lấy dữ liệu của 1000 nến lịch sử
    -   Dự đoán timeframe kế tiếp
    -   Lấy dữ liệu real-time append vô dữ liệu hiện tại

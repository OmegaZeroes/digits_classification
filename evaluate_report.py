import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np

# 1. Import đúng cấu trúc thư mục
from src.models.model import MLP, DigitsClassifier          # Import class MLP
from src.data.dataloader import getDataSet, DataCollator # Import hàm xử lý data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Lấy Test Dataset (Dùng hàm getDataSet để lấy phần test_dataset 20%)
    _, _, test_dataset, _ = getDataSet(root_dir='data') 
    
    # Khởi tạo DataLoader để chạy dữ liệu qua model
    collator = DataCollator()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collator)

    # 3. Khởi tạo và Load Model
    success = False
    while success != True:
        choice = input("Type 1 to use MLP or 2 for CNN: ")
        if int(choice)==1:
            model = MLP().to(device)
            success = True
        elif int(choice)==2:
            model = DigitsClassifier.to(device)
            success =True
        else:
            print("Typo!")
            
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Đang chạy dự đoán trên tập test...")
    
    # 4. Chạy vòng lặp dự đoán
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Lấy chỉ số có xác suất cao nhất (Dự đoán)
            _, predicted = torch.max(outputs, 1)
            
            # Chuyển nhãn One-hot về dạng số (0-9) để sklearn đọc được
            _, targets = torch.max(labels, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # 5. Xuất Classification Report
    target_names = [f"Số {i}" for i in range(10)]
    report = classification_report(all_labels, all_preds, target_names=target_names)

    print("\n--- KẾT QUẢ TỔNG HỢP ---")
    print(report)

    with open('classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("\nĐã lưu báo cáo vào file 'classification_report.txt'")

if __name__ == "__main__":
    main()
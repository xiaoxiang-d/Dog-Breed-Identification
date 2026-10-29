import torch
import torch.nn as nn
from torchvision import models
from scipy.io import loadmat
from pathlib import Path
from torchvision import transforms
from dataset import DogDataset
from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 120)

model.load_state_dict(
    torch.load(
        "results/last_model.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

data_dir = Path("data")

test_data = loadmat(data_dir / "test_list.mat")

test_files = [
    str(item[0][0]) for item in test_data["file_list"]
]

test_labels = [
    int(item[0]) - 1 for item in test_data["labels"]
]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = DogDataset(
    image_dir=data_dir / "Images",
    file_list=test_files,
    labels=test_labels,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total * 100

result_path = Path("results/test_result.txt")

with open(result_path, "w", encoding="utf-8") as f:
    f.write(f"测试图片总数：{total}\n")
    f.write(f"预测正确数量：{correct}\n")
    f.write(f"官方测试集准确率：{test_accuracy:.2f}%\n")

print("测试结果已保存！")
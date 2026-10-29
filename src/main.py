from scipy.io import loadmat
from pathlib import Path
from torchvision import transforms
from dataset import DogDataset
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path("data")

train_data = loadmat(data_dir / "train_list.mat")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet的均值
        std=[0.229, 0.224, 0.225]    # ImageNet的标准差
    )
])

train_files = [
    item[0][0] for item in train_data["file_list"]
]

train_labels = [
    int(item[0]) - 1 for item in train_data["labels"]
]

train_files, val_files, train_labels, val_labels = train_test_split( 
    train_files,
    train_labels,
    test_size=0.2,   # 划分20%作为验证集
    random_state=42,
    stratify=train_labels
)

train_dataset = DogDataset(
    image_dir=data_dir / "Images",
    file_list=train_files,
    labels=train_labels,
    transform=transform
)

val_dataset = DogDataset(
    image_dir=data_dir / "Images",
    file_list=val_files,
    labels=val_labels,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True # 打乱
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False  # 训练时打乱，验证时不打乱
)

images, labels = next(iter(train_loader))

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

for param in model.parameters():
    param.requires_grad = False   # 冻结模型参数

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 120)  # 修改分类层为120个类别

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
images = images.to(device)


optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss()

labels = labels.to(device)

num_epochs = 5
best_accuracy = 0.0

for epoch in range(num_epochs):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
if accuracy > best_accuracy:
    best_accuracy = accuracy

    torch.save(
        model.state_dict(),
        "results/best_model.pth"
    )

    print(f"已保存最佳模型，验证准确率：{best_accuracy:.2f}%")

print(f"验证集准确率：{accuracy:.2f}%")

Path("results").mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    "results/last_model.pth"
)

print("模型保存成功！")
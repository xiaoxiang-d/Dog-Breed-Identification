from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class DogDataset(Dataset):

    def __init__(self,image_dir, file_list, labels, transform=None):
        self.image_dir = image_dir
        self.file_list = file_list
        self.labels = labels
        self.transform = transform # 图像预处理规则，None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_name = self.file_list[index]
        label = self.labels[index]
        image_path = Path(self.image_dir) / image_name
        image = Image.open(image_path).convert("RGB") # 将图像转换为RGB模式
        if self.transform is not None:
            image = self.transform(image)
        return image, label
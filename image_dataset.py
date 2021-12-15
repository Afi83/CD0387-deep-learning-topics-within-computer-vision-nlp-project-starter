import torchvision.transforms as transforms
import torch.utils.data import Dataset
import os
from PIL import Image
# taken and modified class for dataset loading from the following op
# https://stackoverflow.com/questions/54003052/how-do-i-implement-a-pytorch-dataset-for-use-with-aws-sagemaker
class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.all_images = [os.path.join(path, f) for f in os.listdir(path) if os.isfile(os.path.join(path, f)) and f.endswith('.jpg')]
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = self.all_images[idx]

        
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
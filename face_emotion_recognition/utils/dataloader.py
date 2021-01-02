from torch.utils.data import Dataset
from utils.prepocessing import example_set
from PIL import Image
import numpy as np


class EmotionClassificationDataset(Dataset):
    def __init__(self):
        self.example_set, self.class_dict = example_set()
        self.length = len(self.example_set)

    def __getitem__(self, index):
        example_record = self.example_set[index]
        image = np.array(Image.open(example_record[0]).convert('RGB').resize((48, 48)))
        image = image.transpose((2, 0, 1))
        label = example_record[1]

        return (image / 255.).astype('float64'), label

    def __len__(self):
        return self.length

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

from PIL import Image
import os

class TumorDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
        ])
        self.random_transform = {'hflip': TF.hflip,
                                'vflip': TF.vflip,
                                'rotate': TF.rotate}

    def __len__(self):
        total_files = len(os.listdir(self.root))
        return total_files // 2

    def __getitem__(self, index):
        image_name = os.path.join(self.root, str(index)+'.png')
        mask_name = os.path.join(self.root, str(index)+'_mask.png')

        image = Image.open(image_name)
        mask = Image.open(mask_name)

        image = self.transform(image)
        mask = self.transform(mask)

        image, mask = self._random_transform(image, mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        sample = {'index': int(index), 'image': image, 'mask': mask}
        return sample


    def _random_transform(self, image, mask):
        choice_list = list(self.random_transform)
        for _ in range(len(choice_list)):
            choice_key = random.choice(choice_list)

            action_prob = random.randint(0, 1)
            if action_prob >= 0.5:
                if choice_key == 'rotate':
                    rotation = random.randint(15, 75)
                    image = self.random_transform[choice_key](image, rotation)
                    mask = self.random_transform[choice_key](mask, rotation)
                else:
                    image = self.random_transform[choice_key](image)
                    mask = self.random_transform[choice_key](mask)
            choice_list.remove(choice_key)

        return image, mask

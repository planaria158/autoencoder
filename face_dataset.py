
import os
import pandas as pd
import torch
import numpy as np
import random
from torchvision.io import read_image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.noise_factor = 0.50
        filepaths = pd.Series(list(self.img_dir.glob(r'**/*.jpg')), name='Filepath', dtype = 'object').astype(str)
        ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age', dtype = 'object').astype(int)
        self.images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        print('Dataset has', self.images.shape[0], 'rows')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        num_rows = self.images.shape[0]
        done = False
        # filter out anything that is not an rgb image (there are some b&w images in this dataset)
        while not done:
            img_path = self.images.iloc[idx]['Filepath']
            age = self.images.iloc[idx]['Age']
            image = read_image(img_path).to(torch.float32)
            label = age
            if list(image.shape) == [3, 128, 128]:
                done = True 
            else:
                idx = random.randint(0, num_rows-1)

                
        if self.transform:
            image = self.transform(image)

            # TODO: add gaussian noise in follow on experiment for denoising autoencoder
            noisy_image = image + self.noise_factor * torch.rand(image.shape) * 255
            noisy_image = np.clip(noisy_image, 0., 255.)
            noisy_image = (noisy_image/ torch.max(noisy_image)) - 0.5

            # Normalize image centered on 0
            image = (image / torch.max(image)) - 0.5
        if self.target_transform:
            label = self.target_transform(label)

        # return image.float(), image.float()  
        return noisy_image.float(), image.float() 
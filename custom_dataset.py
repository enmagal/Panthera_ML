import pandas as pd
import numpy as np

from skimage.io import imread
from skimage.transform import resize

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

class PantheraDataset:
    def __init__(self, csv_file, root_dir, dataType, transform=None):
        dataset = pd.read_csv(csv_file)
        dataset = dataset.loc[dataset['Type'] == dataType]

        # Update files name in the correct format
        dataset['Photo'] = dataset['Photo'].map(lambda id: '{:08}'.format(id))
        dataset['Photo'] = dataset['Photo'].astype(str)

        nbrImg = len(dataset)
        
        dataset = dataset.reset_index(drop=True) 
        
        self.data = dataset
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path = self.root_dir + str(self.data.loc[idx, 'Photo']) + '.jpg'
        image = imread(image_path)
    
        label = self.data.loc[idx, 'Animal'].astype(np.int64)
        sample={'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class Resize(object):
    """Resize the image in a sample to a given size.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        img = resize(image, (self.output_size, self.output_size))

        return {'image': img, 'label': label}

class Normalize(object):
    """Normalize the image.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        img = image/255.0

        return {'image': img, 'label': label}

class ToTensor(object):
    """Transform data into tensor."""

    def __init__(self):
        self = self
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        image = image.reshape(3, len(image), len(image))
        img = torch.from_numpy(image).float()
        label = torch.from_numpy(np.array(label))

        return {'image': img, 'label': label}

if __name__ == '__main__':
    transform = transforms.Compose([
        # resize
        #Resize(224),
        # normalize
        Normalize(0, 1),
        # to-tensor
        ToTensor()
    ])

    dataset_train = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img/', 'train', transform)
    dataset_test = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img/', 'test', transform)

    print("Label shape : ", dataset_test[8]['label'].shape)

    print("Data example : ", dataset_test[0])

    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=0)
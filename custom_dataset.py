import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

class PantheraDataset:
    def __init__(self, csv_file, root_dir, dataType, transform=None):
        dataset = pd.read_csv(csv_file)
        if dataType != 'all':
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
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
    
        label = self.data.loc[idx, 'Animal'].astype(np.int64)
        label =  torch.from_numpy(np.array(label))
        sample={'image': image, 'label': label}
            
        return sample

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img/', 'train', transform)
    dataset_test = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img/', 'test', transform)

    print("Label shape : ", dataset_test[8]['label'].shape)

    print("Data example : ", dataset_test[0])

    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=0)
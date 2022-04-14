import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from custom_dataset import PantheraDataset

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img_224/', 'all', transform)    

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    
    dfTest = pd.read_csv("./panthera_dataset/list_photo_prep.csv")
    dfTest = dfTest.assign(ResNet18_succeed = True)
    dfTest['Photo'] = dfTest['Photo'].map(lambda id: '{:08}'.format(id))
    dfTest['Photo'] = dfTest['Photo'].astype(str)

    ResNet18 = torch.load('./models/Panthera_ResNet18.pt')
    print(ResNet18.layer4[-1])

    for batch in dataloader:
        x = batch['image'].to(device)
        y = batch['label'].to(device)
        pred = ResNet18(x)
        
    dfTest['ResNet18_succeed'] = (torch.max(pred, 1)[1] == y).numpy()

    dfTest = dfTest.loc[dfTest['ResNet18_succeed'] == False]
    dfTest = dfTest.reset_index(drop=True)

    root_dir = './panthera_dataset/img/'

    for i in range(len(dfTest)):
        image_path = root_dir + str(dfTest.loc[i, 'Photo']) + '.jpg'
        image = Image.open(image_path)
        image.show()
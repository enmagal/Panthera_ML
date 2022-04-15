import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from custom_dataset import PantheraDataset
from ResNet18_GradCAM import ResNet

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

    for batch in dataloader:
        x = batch['image'].to(device)
        y = batch['label'].to(device)
        pred = ResNet18(x)
        
    dfTest['ResNet18_succeed'] = (torch.max(pred, 1)[1] == y).numpy()

    dfTest = dfTest.loc[dfTest['ResNet18_succeed'] == True]
    dfTest = dfTest.reset_index(drop=True)

    dfTest.to_csv('./panthera_dataset/bad_prediction.csv', index=False)

    root_dir = './panthera_dataset/img/'

    dataset = PantheraDataset('./panthera_dataset/bad_prediction.csv', './panthera_dataset/img_224/', 'all', transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    resnet = ResNet()

    idx = 0
    for batch in dataloader:
        # get the image
        img = batch['image']

        # forward pass
        pred = resnet(img)

        pred.argmax(dim=1) # prints tensor([2])

        # get the gradient of the output with respect to the parameters of the model
        pred[:,1].backward()

        # pull the gradients out of the model
        gradients = resnet.get_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = resnet.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        plt.matshow(heatmap.squeeze())

        # make the heatmap to be a numpy array
        heatmap = heatmap.numpy()

        img = cv2.imread('./panthera_dataset/img/' + dfTest.loc[idx, 'Photo'] + '.jpg')
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite('./Results/GradCAM/' + dfTest.loc[idx, 'Photo'] + '.jpg', superimposed_img)

        idx += 1

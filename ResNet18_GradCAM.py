import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from custom_dataset import PantheraDataset

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img_224/', 'all', transform)    

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    ResNet18 = torch.load('./models/Panthera_ResNet18.pt')

    target_layers = [ResNet18.layer4[-1]]
    input_tensor = next(iter(dataloader))['image']
    print(input_tensor.shape)
    
    cam = GradCAM(model=ResNet18, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
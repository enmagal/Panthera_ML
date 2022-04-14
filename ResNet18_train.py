import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from model_train import train
from custom_dataset import PantheraDataset

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.RandomRotation(90),
                                transforms.RandomGrayscale(),
                                transforms.RandomInvert(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAdjustSharpness(2),
                                transforms.RandomAutocontrast(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img_224/', 'train', transform_train)
    dataset_test = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img_224/', 'test', transform_test)

    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=0)

    ResNet18 = models.resnet18(pretrained = True)

    # We freeze the parameters of the ResNet Model
    for param in ResNet18.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = ResNet18.fc.in_features
    ResNet18.fc = nn.Linear(num_ftrs, 2)

    ResNet18 = ResNet18.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(ResNet18.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    history = train(ResNet18, optimizer_conv, loss_fn, train_loader, test_loader, epochs=100, device=device)

    torch.save(ResNet18, "./models/Panthera_ResNet18.pt")

    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(epochs, acc, 'b', label='Training acc')
    ax1.plot(epochs, val_acc, 'r', label='Validation acc')
    ax1.set_title('Training and validation accuracy')
    ax1.legend()

    ax2.plot(epochs, loss, 'b', label='Training loss')
    ax2.plot(epochs, val_loss, 'r', label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.legend()

    plt.show()
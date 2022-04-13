import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from model_train import train
from custom_dataset import PantheraDataset, Resize, Normalize, ToTensor

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        # resize
        #Resize(224),
        # normalize
        Normalize(0, 1),
        # to-tensor
        ToTensor()
    ])

    dataset_train = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img_224/', 'train', transform)
    dataset_test = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img_224/', 'test', transform)

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

    history = train(ResNet18, optimizer_conv, loss_fn, train_loader, test_loader, epochs=25, device=device)

    torch.save(ResNet18, "./models/Panthera_ResNet18.pt")

    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from custom_dataset import PantheraDataset, Resize, Normalize, ToTensor

def visualize_model(model, dataloader, num_images=6, device='cpu'):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    class_names = ['No Animal', 'Animal']

    with torch.no_grad():
        for batch in dataloader:

            x = batch['image'].to(device)
            y = batch['label'].to(device)

            outputs = model(x)
            _, preds = torch.max(outputs, 1)

            for j in range(x.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                plt.imshow(x.cpu().data[j].permute(1, 2, 0))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

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
    dataset = PantheraDataset('./panthera_dataset/list_photo_prep.csv', './panthera_dataset/img_224/', 'train', transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = torch.load('./models/Panthera_ResNet18.pt')
    visualize_model(model, loader, device=device)

    plt.ioff()
    plt.show()
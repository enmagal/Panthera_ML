import pandas as pd
import random

from skimage.io import imread, imsave
from skimage.transform import resize

dfPanthera = pd.read_excel("./panthera_dataset/Liste_photos.xlsx")

# Rename columns

dfPanthera = dfPanthera.rename(columns = {'Nom photo': 'Photo', 'Présence animal': 'Animal', 'Panthère': 'Panthera'})

# Standardize the presence data

dfPanthera.loc[dfPanthera['Animal'] == 'Oui', 'Animal'] = True
dfPanthera.loc[dfPanthera['Animal'] == 'Non', 'Animal'] = False
dfPanthera.loc[dfPanthera['Panthera'] == 'Oui', 'Panthera'] = True
dfPanthera.loc[dfPanthera['Panthera'].isnull(), 'Panthera'] = False

# Update files name in the correct format

dfPanthera['Photo'] = dfPanthera['Photo'].map(lambda id: '{:08}'.format(id))
dfPanthera['Photo'] = dfPanthera['Photo'].astype(str)

# Image resize
nbrImg = len(dfPanthera)
root_dir = "./panthera_dataset/img/"

for i in range(nbrImg):
    image_path = root_dir + str(dfPanthera.loc[i, 'Photo']) + '.jpg'
    image = imread(image_path)
    image = resize(image, (224, 224))
    imsave('./panthera_dataset/img_224/'+ dfPanthera.loc[i, 'Photo'] + '.jpg', image)


# Create to columns for pictures size

dfPanthera = dfPanthera.assign(ImgSizeX = 0)
dfPanthera = dfPanthera.assign(ImgSizeY = 0)

# Split the dataset into train and test data

dfPanthera = dfPanthera.assign(Type = 'train')
testIdx = random.sample(range(nbrImg), int((nbrImg*20)/100))
dfPanthera.loc[testIdx, 'Type'] = 'test'


dfPanthera.to_csv('./panthera_dataset/list_photo_prep.csv', index=False)


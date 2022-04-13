import pandas as pd 
from PIL import Image
import random

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

# Create to columns for pictures size

dfPanthera = dfPanthera.assign(ImgSizeX = 0)
dfPanthera = dfPanthera.assign(ImgSizeY = 0)

nbrImg = len(dfPanthera)

# Compute pictures size

for i in range(nbrImg):
    imgName = dfPanthera.loc[i, 'Photo'] + '.jpg'
    img = Image.open('./panthera_dataset/img/' + imgName)
    dfPanthera.loc[i, 'ImgSizeX'] = img.size[0]
    dfPanthera.loc[i, 'ImgSizeY'] = img.size[1]

# Split the dataset into train and test data

dfPanthera = dfPanthera.assign(Type = 'train')
testIdx = random.sample(range(nbrImg), int((nbrImg*20)/100))
dfPanthera.loc[testIdx, 'Type'] = 'test'


dfPanthera.to_csv('./panthera_dataset/list_photo_prep.csv', index=False)
import os
import torch
import numpy as np
import setup.dataset as dataset
import setup.model as model
import setup.model2 as model2
import setup.classifier as classifier
import setup.plot as plot
from torch.utils.data import SubsetRandomSampler
DATASET_PATH = 'png_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

unet_classifier.model.load_state_dict(torch.load('state_dict_model.pt'))

unet_model.eval()
unet_score = unet_classifier.test(test_loader)
print(f'\n\nDice Score {unet_score}')
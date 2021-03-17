import os
import torch
import numpy as np
import setup.dataset as dataset
import setup.UNet as UNet
import setup.ResUNet as ResUNet
import setup.classifier as classifier
import setup.plot as plot
from torch.utils.data import SubsetRandomSampler

np.random.seed(0)
torch.manual_seed(0)

DATASET_PATH = '/home/tungdao/Tung/code/ducanh/data/png_dataset'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def sampler_indices(length):
    indices = list(range(length))
    np.random.shuffle(indices)
    split = int(np.floor(0.1 * length))
    train_indices, test_indices = indices[split:], indices[:split]
    return train_indices, test_indices

tumor_dataset = dataset.TumorDataset(DATASET_PATH)

train_indices, test_indices = sampler_indices(len(tumor_dataset))
train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(tumor_dataset, batch_size=1, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(tumor_dataset, batch_size=1, sampler=test_sampler)

FILTER_LIST = [16,32,64,128,256]
#unet_model = UNet.DynamicUNet(FILTER_LIST).to(device)
unet_model = ResUNet.ResUNet(FILTER_LIST).to(device)
unet_classifier = classifier.TumorClassifier(unet_model, device)

unet_classifier.model.load_state_dict(torch.load('ResUNet.pt'))

unet_model.eval()
unet_score = unet_classifier.test(test_loader)
print(f'\n\nDice Score {unet_score}')

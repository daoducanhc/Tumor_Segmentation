import os
import torch
import numpy as np
import setup.dataset as dataset
import setup.model as model
import setup.model2 as model2
import setup.classifier as classifier
import setup.plot as plot
from torch.utils.data import SubsetRandomSampler

np.random.seed(0)
torch.manual_seed(0)

DATASET_PATH = '/home/tungdao/Tung/code/ducanh/data/png_dataset'
# DATASET_PATH = 'png_dataset'
BATCH_SIZE = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def sampler_indices(length):
    indices = list(range(length))
    np.random.shuffle(indices)
    index1 = int(np.floor(0.1 * length))
    index2 = int(np.floor(0.2 * length))
    test_indices, valid_indices, train_indices = indices[:index1], \
                                                indices[index1:index2], \
                                                indices[index2:] \
    return train_indices, valid_indices, test_indices

tumor_dataset = dataset.TumorDataset(DATASET_PATH)

train_indices, valid_indices, test_indices = sampler_indices(len(tumor_dataset))
train_sampler, valid_sampler, test_sampler = SubsetRandomSampler(train_indices), \
                                            SubsetRandomSampler(valid_indices), \
                                            SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(tumor_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(tumor_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(tumor_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

FILTER_LIST = [16,32,64,128,256]
unet_model = model.DynamicUNet(FILTER_LIST).to(device)
# unet_model = model2.ONet(FILTER_LIST).to(device)
unet_classifier = classifier.TumorClassifier(unet_model, device)

unet_model.train()
unet_classifier.train(train_loader, valid_loader, learning_rate=0.001, epochs=100, name='state_dict_model.pt')

unet_model.eval()
unet_score = unet_classifier.test(test_loader)
print(f'\n\nDice Score {unet_score}')
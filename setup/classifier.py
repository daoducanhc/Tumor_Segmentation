import torch
from .loss import DiceBCELoss
import torch.optim as optim
import numpy as np
import time

class TumorClassifier():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = DiceBCELoss()

    def train(self, trainLoader, validLoader, learning_rate=0.001, epochs=20, name="state_dict_model.pt"):
        last_loss = 1000

        dataLoader = {
            'train': trainLoader,
            'valid': validLoader
        }

        history = {
            'train': list(),
            'valid': list()
        }

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.2)
        print('Starting...')

        for epoch in range(epochs):

            print("\nEpoch {}/{}:".format(epoch+1, epochs))
            epoch_time = time.time()
            # self.epoch_fit(trainLoader, validLoader)
            
            for phase in ['train', 'valid']:
                epoch_loss, iteration = 0, 0

                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()

                for data in dataLoader[phase]:
                    iteration+=1
                    image = data['image'].to(self.device)
                    mask = data['mask'].to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(image)

                        loss_val = self.criterion(output, mask)
                        if phase == 'train':
                            loss_val.backward()
                            self.optimizer.step()

                    epoch_loss += loss_val.item()
                    
                    
                epoch_loss /= (iteration * dataLoader[phase].batch_size)
                history[phase].append(epoch_loss)

                print('{} Loss:{:.7f}'.format(phase, epoch_loss))
                if phase == 'valid' and last_loss > epoch_loss:
                    if last_loss != 1000:
                        torch.save(self.model.state_dict(), name)
                        print('Saved')
                    last_loss = epoch_loss

            end = time.time() - epoch_time
            m = end//60
            s = end - m*60
            print("Time {:.0f}m {:.0f}s".format(m, s))
        return history


    def test(self, testLoader, threshold=0.5):
        self.model.eval()
        test_data_indexes = testLoader.sampler.indices[:]
        data_len = len(test_data_indexes)
        mean_val_score = 0

        batch_size = testLoader.batch_size
        if batch_size != 1:
            raise Exception("Set batch size to 1 for testing purpose")
        testLoader = iter(testLoader)
        while len(test_data_indexes) != 0:
            data = testLoader.next()
            index = int(data['index'])
            if index in test_data_indexes:
                test_data_indexes.remove(index)
            else:
                continue
            image = data['image'].view((1, 1, 512, 512)).to(self.device)
            mask = data['mask']

            mask_pred = self.model(image).cpu()
            mask_pred = (mask_pred > threshold)
            mask_pred = mask_pred.numpy()

            mask = np.resize(mask, (1, 512, 512))
            mask_pred = np.resize(mask_pred, (1, 512, 512))

            mean_val_score += self._dice_coefficient(mask_pred, mask)

        mean_val_score = mean_val_score / data_len
        return mean_val_score

    def _dice_coefficient(self, predicted, target):
        smooth = 1
        product = np.multiply(predicted, target)
        intersection = np.sum(product)
        coefficient = (2*intersection + smooth) / (np.sum(predicted) + np.sum(target) + smooth)
        return coefficient

    def predict(self, data, threshold=0.5):
        self.model.eval()
        image = data['image'].numpy()
        mask = data['mask'].numpy()

        image_tensor = torch.Tensor(data['image'])
        image_tensor = image_tensor.view((-1, 1, 512, 512)).to(self.device)
        output = self.model(image_tensor).detach().cpu()
        output = (output > threshold)
        output = output.numpy()

        image = np.resize(image, (512, 512))
        mask = np.resize(mask, (512, 512))
        output = np.resize(output, (512, 512))
        score = self._dice_coefficient(output, mask)
        return image, mask, output, score

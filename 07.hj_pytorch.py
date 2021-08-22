import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

training_set_path = "training_sample"
validate_set_path = "validate_sample"


class MnistDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(os.path.join(path, f'{path}.csv'))
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx, 0]

        img_path = os.path.join(self.path, f'{idx:05}.png')
        img = imageio.imread(img_path)
        img = img.flatten()
        img_data = torch.FloatTensor(img) / 255.0

        target = torch.zeros(3)
        target[int(label)] = 1.0

        return label, img_data, target

    def plot_images(self, idx):
        plt.title(f'label = {self.df.iloc[idx, 0]}')
        img_path = os.path.join(self.path, f'{idx:05}.png')
        img = imageio.imread(img_path).reshape(200, 200)
        plt.imshow(img, cmap='Blues')
        plt.show()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(200*200, 500),
            nn.Sigmoid(),
            nn.Linear(500, 3),
            nn.Sigmoid()
        )

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []

    def fit(self, inputs, targets):
        outputs = self.forward(inputs)

        self.loss = self.loss_function(outputs, targets)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.counter += 1
        if self.counter % 10000 == 0:
            print(f'counter = {self.counter}')

        if self.counter % 10 == 0:
            self.progress.append(self.loss.item())


    def forward(self, inputs):
        return self.model(inputs)




mnist_train_set = MnistDataset(training_set_path)
mnist_valid_set = MnistDataset(validate_set_path)
C = Classifier()

# mnist_train_set.plot_images(10)


epochs = 1
for idx in range(epochs):
    for label, inputs, targets in mnist_train_set:
        C.fit(inputs, targets)



# outputs = C.forward(mnist_valid_set[10])


score = 0
items = 0
for label, inputs, targets in mnist_valid_set:
    items += 1
    outputs = C.forward(inputs)
    if outputs.argmax() == label:
        score += 1

print(f'acc: {score/items:.2f}')
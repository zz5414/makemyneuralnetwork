import pandas as pd
import matplotlib.pyplot as plt
#https://www.delftstack.com/ko/howto/matplotlib/how-to-draw-rectangle-on-image-in-matplotlib/
import matplotlib.patches as patches
import imageio
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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
        lt = eval(self.df.loc[idx, ['lt']].values[0])
        rb = eval(self.df.loc[idx, ['rb']].values[0])
        target_bbox = torch.FloatTensor((lt[0], lt[1], rb[0], rb[1]))

        img_path = os.path.join(self.path, f'{idx:05}.png')
        img = imageio.imread(img_path)
        img = img.flatten()
        img_data = torch.FloatTensor(img) / 255.0

        target_class = torch.zeros(3)
        target_class[int(label)] = 1.0

        return label, img_data, target_class, target_bbox

    def plot_images(self, idx):
        plt.title(f'label = {self.df.iloc[idx, 0]}')
        lt = eval(self.df.loc[idx, ['lt']].values[0])
        rb = eval(self.df.loc[idx, ['rb']].values[0])

        img_path = os.path.join(self.path, f'{idx:05}.png')

        img = imageio.imread(img_path).reshape(200, 200)
        plt.imshow(img, cmap='Blues')
        ax = plt.gca()

        rect = patches.Rectangle(lt, rb[0] - lt[0], rb[1] - lt[1], fill=False, edgecolor='red')
        ax.add_patch(rect)

        plt.show()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(200*200, 500),
            nn.Sigmoid(),
        )

        self.fig_type_classifier = nn.Linear(500, 3)
        self.bbox_regressor = nn.Linear(500, 4)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []

    def fit(self, inputs, targets, target_bbox):
        pred_class, bbox = self.forward(inputs)

        loss_f = nn.MSELoss()
        loss_class = loss_f(pred_class, targets)

        loss_b = nn.MSELoss()
        #loss_bb = F.l1_loss(bbox, target_bbox, reduction='none').sum(1)
        #loss_bb = loss_bb.sum()
        loss_bb = loss_b(bbox, target_bbox)

        self.loss = loss_class + loss_bb

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.counter += 1
        if self.counter % 100 == 0:
            print(f'counter = {self.counter}')

        if self.counter % 10 == 0:
            self.progress.append(self.loss.item())


    def forward(self, inputs):
        output = self.model(inputs)
        pred_class = self.fig_type_classifier(output)
        bbox = self.bbox_regressor(output)
        return pred_class, bbox

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()

    def draw_prediction(self, inputs, targets, target_bbox):
        pred_class, out_bbox = self.forward(inputs)
        print(f'oub_bbox = {out_bbox}')


        plt.title(f'pred_label = {pred_class.argmax()}')

        img = inputs.reshape(200, 200)
        plt.imshow(img, cmap='Blues')
        ax = plt.gca()


        out_bbox = out_bbox[0]
        if out_bbox[0] > 0 and \
            out_bbox[1] > 0 and \
            out_bbox[2] > 0 and \
            out_bbox[3] > 0:
            rect = patches.Rectangle((out_bbox[0], out_bbox[1]), out_bbox[2] - out_bbox[0], out_bbox[3] - out_bbox[1], fill=False, edgecolor='red')
            ax.add_patch(rect)

        plt.show()


def print_acc(valid_set, C):
    score = 0
    items = 0
    for label, inputs, targets, target_bbox in valid_set:
        items += 1
        outputs, bbox = C.forward(inputs)
        if outputs.argmax() == label:
            score += 1

    print(f'acc: {score / items:.2f}')


mnist_train_set = MnistDataset(training_set_path)
mnist_valid_set = MnistDataset(validate_set_path)
C = Classifier()

# mnist_train_set.plot_images(300)

dataloader = DataLoader(mnist_train_set, batch_size=1, shuffle=True)

# for debugging
#for label, inputs, targets, target_bbox in mnist_valid_set:
#    C.draw_prediction(inputs, targets, target_bbox)

start = time.time()
epochs = 2
for idx in range(epochs):
    for label, inputs, targets, target_bbox in dataloader:
        C.fit(inputs, targets, target_bbox)
    print_acc(mnist_valid_set, C)

C.plot_progress()
print(f'{time.time() - start:.2f}sec')

valid_set_dataloader = DataLoader(mnist_valid_set, batch_size=1, shuffle=True)
for label, inputs, targets, target_bbox in valid_set_dataloader:
    C.draw_prediction(inputs, targets, target_bbox)


import pandas as pd
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# df = pd.read_csv(r"D:\03.Dev\Python\make_your_neural_network\mnist_train.csv", header=None)
# print(df.head())
# print(df.info())
#
# row = 16
# data = df.iloc[row]
# label = data[0]
# img = data[1:].values.reshape(28,28)
#
# plt.title(f'label={label}')
# plt.imshow(img, interpolation='none', cmap='Blues')
# plt.show()


class MnistDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path, header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index, 0]
        target = torch.zeros(10)
        target[label] = 1.0

        image_values = torch.FloatTensor(self.df.iloc[index, 1:].values) / 255.0

        return label, image_values, target

    def plot_image(self, idx):
        data = self.df.iloc[idx]
        label = data[0]
        img = data[1:].values.reshape(28,28)

        plt.title(f'label={label}')
        plt.imshow(img)
        plt.show()



class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.Sigmoid(),

            nn.Linear(200, 10),
            nn.Sigmoid()
        )

        # MSELoss 또한 Module의 자식 클래스일 뿐이다.
        # 그래서 nn.MSELoss가 아니라 nn.MSELoss() 형태로 객체를 만들어주는 것
        # 함수호출인줄 알고 nn.MSELoss로 해야 하는 줄 알았음
        self.loss_function = nn.MSELoss()

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []


    def forward(self, inputs):
        return self.model(inputs)


    def fit(self, inputs, targets):
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        if self.counter % 10000 == 0:
            print(f'counter = {self.counter}')

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()



mnist_dataset = MnistDataset(r"D:\03.Dev\Python\make_your_neural_network\mnist_train.csv")
# mnist_dataset.plot_image(13)
C = Classifier()

epochs = 4
start = time.time()
for idx in range(epochs):
    print(f'training epochs : {idx+1} / {epochs}')
    for label, image_data_tensor, target_tensor in mnist_dataset:
        C.fit(image_data_tensor, target_tensor)
print(f'{time.time() - start:.2f} sec')
C.plot_progress()



mnist_test_dataset = MnistDataset(r"D:\03.Dev\Python\make_your_neural_network\mnist_test.csv")
mnist_test_dataset.plot_image(19)
img_data = mnist_test_dataset[19][1]
output = C.forward(img_data)
pd.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0,1))
plt.show()


score = 0
items = 0

for label, img_data_tensor, target_tensor in mnist_test_dataset:
    answer = C.forward(img_data_tensor).detach().numpy()
    if(answer.argmax() == label):
        score += 1

    items += 1

print(f'scores : {score/items:.2f}')













# row = 13
# data = df.iloc[row]




#
# label = data[0]
# img_data = data[1:].values.reshape(28, 28)
# plt.title(f'label = {label}')
# plt.imshow(img_data, interpolation='none', cmap='Blues')
# plt.show()

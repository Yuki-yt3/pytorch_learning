import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

raw_data = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
train_data, test_data = train_test_split(raw_data, test_size=0.3)


class DiabetesDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.from_numpy(data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_data = DiabetesDataset(train_data)
train_loader = DataLoader(dataset=train_data,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)

test_data = DiabetesDataset(test_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train():
    train_loss=0.0
    for i, data in enumerate(train_loader, 0):
        # prepare data
        inputs, labels = data
        # forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()
        train_loss += loss.item()
    return train_loss


def test():
    with torch.no_grad():
        for i,data in enumerate(test_data,0):
            x_test,y_test =data
        y_pred = model(x_test)
        y_pred_label = torch.where(y_pred>0.5,torch.tensor([1.0]),torch.tensor([0.0]))
        acc = torch.eq(y_pred_label,y_test).sum().item()/y_test.size(0)
        print("test acc:",acc)

if __name__ == '__main__':
    for epoch in range(50000):
        train_loss = train()
        if epoch%2000==0:
            print(epoch,train_loss)
            test()


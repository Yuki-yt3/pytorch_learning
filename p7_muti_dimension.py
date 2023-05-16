import torch.nn
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
print(x_data[0])
print(y_data[0])


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
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%100==0:
        y_pred_val=torch.where(y_pred >= 0.5,torch.tensor([1.0]),torch.tensor([0.0]))
        acc=torch.eq(y_pred_val,y_data).sum().item()/y_data.size(0)
        print('loss=',loss.item(),'acc=',acc)
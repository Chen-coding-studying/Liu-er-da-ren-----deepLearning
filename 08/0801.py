import numpy as np
import torch
import os
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader #utils 工具包 Dataset 抽象类 DataLoader 数据加载类
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class DiabetesDataset(Dataset): #继承Dataset抽象类
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.y_data = (self.y_data > 120).float()
        self.len = xy.shape[0]# 数组的大小
     
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
dataset = DiabetesDataset('diabetes_data.csv')
train_loader = DataLoader(dataset=dataset, 
                          batch_size=32, 
                          shuffle=True, 
                          num_workers=2)   


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    
model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':

    for epoch in range(100):
        for i, data in enumerate(train_loader, 0): 
            
            inputs, labels = data
            
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

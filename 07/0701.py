import torch
import numpy as np
import os 

os.chdir(os.path.dirname(os.path.abspath(__file__)))
xy = np.loadtxt('diabetes_data.csv', dtype = np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])
y_data_binary = (y_data > 120).float()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))   
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data_binary)
    
    print(epoch + 1,loss.item())
    
    print(model.linear1.weight)
    print(model.linear2.weight)
    print(model.linear3.weight)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


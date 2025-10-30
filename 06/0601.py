import torch

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #类似矩阵乘法 y = Wx + b

    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred
    
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[0], [0], [1]])

model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y.float())  

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print(f'\fweights: {model.linear.weight.item()}, bias: {model.linear.bias.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




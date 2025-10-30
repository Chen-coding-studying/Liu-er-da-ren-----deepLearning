import torch

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    
model = LinearModel()
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'W: {model.linear.weight.item()}, b: {model.linear.bias.item()}')
print('Prediction for input 4.0:', model(torch.tensor([[4.0]])).item())
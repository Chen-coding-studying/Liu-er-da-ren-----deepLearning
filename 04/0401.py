import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0], requires_grad=True)

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y).pow(2).sum()

print("Predict (before training)", 4, forward(4).item())

for epoch in range(10):
    for x, y in zip(x_data, y_data): 
        
        l = loss(x,y)
        l.backward()
        print('grad:', x, y, w.grad.item())
        with torch.no_grad():
            w -= 0.01 * w.grad
        
        w.grad.zero_()
        print(f"epoch: {epoch}, loss: {l.item():.4f}")

print("Predict (after training)", 4, forward(4).item())
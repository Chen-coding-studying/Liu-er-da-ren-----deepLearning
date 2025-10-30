import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([1.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

def forward(x):
    return x * w1 + x * w2 + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y).pow(2).sum()

print("Predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data, y_data):
        
        l = loss(x, y)
        l.backward()
        print('grad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        
        with torch.no_grad():
            w1 -= 0.01 * w1.grad
            w2 -= 0.01 * w2.grad
            b -= 0.01 * b.grad
        
        w1.grad.zero_()
        w2.grad.zero_() 
        b.grad.zero_()

        print(f"epoch: {epoch + 1}, loss: {l.item():.4f}")

print("Predict (after training)", 4, forward(4).item())
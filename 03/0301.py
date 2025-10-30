x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  

def forward(x):
    return x * w

def cost(x, y):
    cost = 0
    
    for x_i, y_i in zip(x, y):
        y_pred = forward(x_i)
        cost += (y_pred - y_i) ** 2
    return cost / len(x)    

def gradient(x, y):
    grad = 0
    N = len(x)
    for x_i, y_i in zip(x, y):
        y_pred = forward(x_i)
        grad += 2 * x_i * (y_pred - y_i)
    return grad / N

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print(f'Epoch {epoch}: w = {w:.4f}, cost = {cost_val:.4f}')

print(f'Final parameter: w = {w:.4f}')  
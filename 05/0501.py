import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):# 不用手写前向传播
    def __init__(self):
        super(LinearModel, self).__init__()# 调用父类 并初始化 
        self.linear = torch.nn.Linear(1, 1)# 构造对象

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

'''
nn.Module 重写了 __call__ 方法，调用对象时会自动调用 forward 方法
所以可以直接通过 model(x) 来调用 forward 方法

'''

model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)# 构造 MSEloss 均方误差损失函数, 不对损失取平均
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)# 构造优化器对象 |SGD 随机梯度下降

'''
新版本size_average已弃用，建议使用reduction参数
reduction='sum' 对应 size_average=False

'''

for epoch in range(200):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch+1}, Loss: {loss}')

    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度   
    optimizer.step()       # 更新参数  

print(f'W: {model.linear.weight.item()}, b: {model.linear.bias.item()}')

x_test = torch.tensor([[4.0]])
y_test = model(x_test)

print(f'Prediction for input 4.0: {y_test.item()}')

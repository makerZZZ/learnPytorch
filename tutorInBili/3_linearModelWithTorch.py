import torch
import torch.nn as nn

x_data = torch.Tensor([[1.], [2.], [3.]])
y_data = torch.Tensor([[2.], [4.], [6.]])

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch: ', epoch, 'loss: ', loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.]])
y_test = model(x_test)
print('y_pred = ', y_test.item())

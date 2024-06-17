import torch
import matplotlib.pyplot as plt 

x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w = torch.tensor([1.], requires_grad=True)

def forward(x):
    y_pred = x * w
    return y_pred
    
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

epoch_list = []
loss_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        loss_val.backward()
        # print('grad: ', x, y, w.grad.item())
        print("w = ", w)
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    epoch_list.append(epoch)
    loss_list.append(loss_val.item())
    print('process:', epoch + 1, loss_val.item())

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('2_png')
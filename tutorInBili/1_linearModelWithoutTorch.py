import numpy
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w = 1.

def forward(x):
    return x * w

def loss(x_data, y_data):
    loss_sum = 0
    for x, y in zip(x_data, y_data):
        y_pred = forward(x)
        loss_sum += (y_pred - y) ** 2
    return loss_sum / len(x_data)

def gradient(x_data, y_data):
    gradient_sum = 0
    for x, y in zip(x_data, y_data):
        y_pred = forward(x)
        gradient = 2 * (y_pred - y) * x
        gradient_sum += gradient
    return gradient_sum / len(x_data)

epoch_list = []
loss_list = []
for epoch in range(100):
    loss_val =loss(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w = w - 0.01 * grad_val
    print(f"epoch: {epoch}\n w: {w}, loss: {loss_val}")
    epoch_list.append(epoch + 1)
    loss_list.append(loss_val)
    
plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig('1_png')
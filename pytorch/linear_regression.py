import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

# 生成数据
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32).reshape(-1, 1)
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)

# 数据标准化
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out  

input_size = 1    
output_size = 1
model = LinearRegressionModel(input_size, output_size)

model.to(device)

# 训练参数
epoches = 5000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练模型
for epoch in range(epoches):
    epoch += 1

    # tensor from numpy
    input = torch.from_numpy(x_train_scaled).to(device)
    labels = torch.from_numpy(y_train_scaled).to(device)

    # zero grad
    optimizer.zero_grad()

    # forward pass
    output = model(input)

    # loss
    loss = criterion(output, labels)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # print loss
    if epoch % 500 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

# 测试模型
test = [i for i in range(10, 21)]
test = np.array(test, dtype=np.float32).reshape(-1, 1)
test_scaled = scaler_x.transform(test)

# Move test data to cuda
test_tensor = torch.from_numpy(test_scaled).to(device)
prediction_scaled = model(test_tensor).data.cpu().numpy()  # Move prediction back to cpu for inverse transform
prediction = scaler_y.inverse_transform(prediction_scaled)

print("Test Input:")
print(test)
print("Predicted Output:")
print(prediction)
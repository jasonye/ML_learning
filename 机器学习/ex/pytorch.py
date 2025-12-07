import torch

# 当前安装的 PyTorch 库的版本
print(torch.__version__)
# 检查 CUDA 是否可用，即你的系统有 NVIDIA 的 GPU
print(torch.cuda.is_available())

x = torch.rand(5, 3)
print(x)

# 张量相加
e = torch.rand(2, 3)
f = torch.ones(2, 3)
print("e:", e)
print("e.T", e.T)

print("e.shape:", e.shape)
print(e.t())

print("f:", f)


print(e + f)

# 逐元素乘法
print(e * f)
print(f.dot(e.t()))  # 矩阵乘法
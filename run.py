import torch
import time

device = "cuda:0"

# 想占多少GB显存
memory_gb = 40

# float32 每个元素4bytes
num_elements = memory_gb * 1024**3 // 4

# 占显存
holder = torch.empty(num_elements, dtype=torch.float32, device=device)

print(f"Allocated about {memory_gb} GB on {device}")

# 一个小tensor用于偶尔计算
a = torch.randn((256, 256), device=device)
b = torch.randn((256, 256), device=device)

while True:
    # 偶尔跑一个很小的计算
    c = torch.matmul(a, b)

    # 同步一下GPU
    torch.cuda.synchronize()

    print("heartbeat - GPU alive")

    # 长时间休眠
    time.sleep(60)
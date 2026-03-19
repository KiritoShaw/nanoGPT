# PyTorch

## compile

torch.compile 是 PyTorch 2.0 引入的一个核心功能，是一个强大的 JIT 编译器，旨在通过 `TorchDynamo`、`TorchInductor` 等工具链，自动化地将 PyTorch 代码优化为高效的硬件内核

```python
# 装饰器形式
@torch.compile
def func(x):
    return torch.sin(x)

# 函数调用形式
model = torch.nn.Linear(100, 10)
optimized_model = torch.compile(model)
```

## DDP 

Distributed Data Parallel 的缩写，是 PyTorch 中用于多 GPU 分布式训练的核心模块。它通过将模型复制到多个设备（GPU）上，每个设备处理不同的数据子集，并在反向传播后同步梯度来高效地实现并行训练

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 初始化进程组
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 模型放置到对应 GPU
model = MyModel().cuda(local_rank)
ddp_model = DDP(model, device_ids=[local_rank])

# 数据加载器使用 DistributedSampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

# 训练循环
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # 确保每个 epoch 数据打乱不同
    for data, target in dataloader:
        data, target = data.cuda(local_rank), target.cuda(local_rank)
        output = ddp_model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## TF32 vs FP32

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**TF32**（TensorFloat32）：利用 **Tensor Cores**（张量核心）加速，这些专用单元能在单个时钟周期内完成 4x4 矩阵乘法累加。在支持 TF32 的 NVIDIA Ampere 架构及更新的 GPU（如 A100、H100、RTX 30/40 系列）上，TF32 能提供接近 FP32 的动态范围，尾数位少，精度略低，但速度更快（通常与 FP16 相当），可达到 FP32 的 8~16 倍

> 一个浮点数通常由三部分组成：
>
> - **符号位**：表示正负（0 为正，1 为负）
> - **指数位**：表示 2 的多少次幂，相当于科学记数法中的指数部分
> - **尾数位**：用来存储“有效数字”的二进制形式
>
> FP32（单精度浮点数）为例，它用 23 位 来存储尾数；TF32 只用了 10 位 来存储尾数

## np.memmap

**`np.memmap`** 是 NumPy 提供的一种“内存映射”机制。它把磁盘上的大文件映射到虚拟内存中，让你可以像操作数组一样访问文件内容，但**不会一次性把整个文件读入物理内存**

```python
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
```

每个 batch 重新创建 memmap 是为了避免内存泄漏。在 Windows 系统上或某些情况下，反复使用同一个 `memmap` 对象进行读取，可能会导致内存使用量不断增加（因为底层资源没有被及时释放）。每次调用 `get_batch` 都重新打开文件，可以确保每次用完文件后，资源被正确回收，内存不会膨胀

## 异步数据传输

`.pin_memory()`：将 CPU 张量锁定在固定的内存页中（page-locked memory）。这样 GPU 就可以通过 DMA（直接内存访问）从这块内存拷贝数据，速度更快，并且可以**异步**进行

`.to(device, non_blocking=True)`：告诉 PyTorch 启动一个异步拷贝操作，把数据从 CPU 复制到 GPU，同时不阻塞当前 Python 线程。这样，CPU  可以继续做其他事情（比如准备下一个 batch），而拷贝操作在后台进行。当模型需要用到这些数据时，PyTorch  会自动等待拷贝完成（如果还没完成）

```python
if device_type == 'cuda':
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
else:
    x, y = x.to(device), y.to(device)
```


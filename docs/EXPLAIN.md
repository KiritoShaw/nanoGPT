# nanoGPT 关键模块设计说明

本文档按“数据准备 -> 配置系统 -> 模型定义 -> 训练循环 -> 推理采样”的顺序，解释 nanoGPT 的关键模块是如何设计的，并附上对应代码片段。整体上，这个仓库最核心的设计取向有两个：

1. 尽量把主干逻辑放在少量文件里，便于直接修改。
2. 在保持代码短小的前提下，保留训练 GPT 所需的关键能力，例如 DDP、AMP、检查点、预训练权重加载和采样推理。

## 1. 总体结构

可以先把仓库理解成下面这条主链路：

```text
原始文本
  -> data/*/prepare.py
  -> train.bin / val.bin / meta.pkl
  -> train.py
  -> model.py
  -> out/ckpt.pt
  -> sample.py
  -> 生成文本
```

各模块职责如下：

- `data/*/prepare.py`：把原始文本转换成连续 token id，并保存为二进制文件。
- `configurator.py`：让 `train.py` 和 `sample.py` 能被配置文件或命令行快速覆盖。
- `model.py`：定义 GPT 的网络结构、优化器分组、预训练权重加载和生成逻辑。
- `train.py`：完成训练初始化、数据读取、评估、梯度更新、学习率调度和检查点保存。
- `sample.py`：加载训练好的模型或 GPT-2 权重，编码 prompt 并逐步采样输出。
- `bench.py`：复用训练核心路径做性能基准测试。

## 2. 数据准备模块

### 2.1 设计目标

数据预处理脚本的目标不是做复杂的数据管道，而是把文本尽快变成训练脚本可以直接读取的紧凑格式。这个设计与 `train.py` 中的 `np.memmap` 是配套的，因此最终产物统一是：

- `train.bin`
- `val.bin`
- `meta.pkl`

其中：

- `train.bin` / `val.bin` 保存连续 token id 流。
- `meta.pkl` 保存词表大小，以及字符级任务需要的 `stoi` / `itos` 映射。

### 2.2 字符级数据集的设计

`data/shakespeare_char/prepare.py` 选择了最简单的字符级编码方案。它的设计思路是：

1. 扫描全文得到唯一字符集合。
2. 构造字符到整数的映射。
3. 按 9:1 切分训练集和验证集。
4. 写出二进制 token 文件和元信息。

对应代码如下：

```python
chars = sorted(list(set(data)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
```

这样设计的好处是非常直观，适合教学和小规模实验。代价是词表粒度太细，序列会更长，表达能力也不如 BPE。

### 2.3 OpenWebText 的设计

`data/openwebtext/prepare.py` 则走的是 GPT-2 BPE 路线。它的设计重点是：利用 HuggingFace `datasets` 做并行切分和映射，再把所有样本拼接成一个大数组，以适配训练时的随机窗口采样。

关键代码如下：

```python
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)
```

写文件时没有保存成逐条样本，而是把所有 id 直接顺序拼接到一个 `memmap` 中：

```python
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
    arr_batch = np.concatenate(batch['ids'])
    arr[idx : idx + len(arr_batch)] = arr_batch
    idx += len(arr_batch)
arr.flush()
```

这样的设计与语言模型训练高度匹配，因为训练并不需要“样本对象”，而是需要“连续 token 流中的随机切片”。

## 3. 配置系统模块

### 3.1 设计思路

`configurator.py` 是这个仓库非常有代表性的设计：它没有引入复杂的配置框架，而是直接通过 `exec(open('configurator.py').read())` 修改调用脚本的全局变量。

这一层的核心目标只有一个：让 `train.py` / `sample.py` 既有默认值，又能被配置文件和命令行快速覆盖。

关键代码如下：

```python
for arg in sys.argv[1:]:
    if '=' not in arg:
        config_file = arg
        exec(open(config_file).read())
    else:
        key, val = arg.split('=')
        key = key[2:]
        attempt = literal_eval(val)
        globals()[key] = attempt
```

这个设计的优点：

- 非常短，几乎零学习成本。
- `train.py` 中的超参数直接就是普通 Python 变量，可读性高。
- 覆盖顺序明确：默认值 -> 配置文件 -> 命令行。

这个设计的代价：

- 依赖 `exec`，灵活但不够严格。
- 类型校验比较弱，只做了基本的 `literal_eval` 和类型一致性检查。
- 不适合大型工程，但非常适合“想快速改实验脚本”的场景。

## 4. 模型定义模块 `model.py`

`model.py` 把完整 GPT 模型集中在一个文件中，这是 nanoGPT 最核心的设计之一。它追求的不是最大抽象层次，而是“打开一个文件就能看懂模型主干”。

### 4.1 LayerNorm：支持可选 bias

PyTorch 默认没有直接提供 `bias=False` 的 LayerNorm 写法，所以仓库自己包了一层：

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

这样做的目的，是让模型能够统一控制“所有线性层和归一化层是否带 bias”。

### 4.2 CausalSelfAttention：把性能优化保留在最少代码里

注意力模块的设计点主要有三个：

1. 用一个线性层一次性投影出 Q、K、V，减少模块数量。
2. 优先使用 PyTorch 2 的 Flash Attention。
3. 如果环境不支持，就退回到手写 masked attention。

核心代码如下：

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

前向传播里有清晰的“双路径”设计：

```python
if self.flash:
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=self.dropout if self.training else 0,
        is_causal=True
    )
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
```

这段实现说明 nanoGPT 的设计理念不是“只写最原始的教学代码”，而是“在可读的前提下，尽量吃到现代 PyTorch 的性能红利”。

### 4.3 Block：Pre-LN 残差结构

每个 Transformer Block 都由两部分组成：

- 注意力子层
- MLP 前馈子层

采用的是 Pre-LN 写法：

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

这种结构的优点是稳定、主流、容易和 GPT-2 对齐。

### 4.4 GPTConfig：把结构超参数集中声明

`GPTConfig` 用 dataclass 保存模型形状：

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```

它不是复杂的配置系统，只是一个轻量的数据容器，让模型初始化、权重恢复和预训练加载都能共享同一套结构描述。

### 4.5 GPT 主体：模块装配和权重绑定

GPT 主类把 token embedding、position embedding、多个 Block 和最终归一化统一装进 `ModuleDict` 中，再定义输出头：

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = LayerNorm(config.n_embd, bias=config.bias),
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
self.transformer.wte.weight = self.lm_head.weight
```

这里最关键的一点是 weight tying，也就是输入 embedding 和输出投影共享参数。这样做的好处是：

- 减少参数量。
- 与 GPT-2 实现保持一致。
- 往往能带来更好的语言建模效果。

### 4.6 forward：训练和推理共用一套主干

`forward` 的整体结构非常清楚：嵌入 -> 多层 Block -> 最终 LayerNorm -> 输出 logits。

```python
tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
for block in self.transformer.h:
    x = block(x)
x = self.transformer.ln_f(x)
```

训练态和推理态在最后一步分叉：

```python
if targets is not None:
    logits = self.lm_head(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1
    )
else:
    logits = self.lm_head(x[:, [-1], :])
    loss = None
```

这个分支设计很实用：

- 训练时保留全序列 logits，便于计算交叉熵。
- 推理时只取最后一个时间步，减少无意义计算。

### 4.7 `from_pretrained`：兼容 HuggingFace GPT-2 权重

这个方法让 nanoGPT 可以直接加载 OpenAI GPT-2 系列权重。设计重点在于：

1. 先按对应结构创建本地 `GPT`。
2. 再从 HuggingFace 模型读取状态字典。
3. 把 GPT-2 中 Conv1D 风格的权重转置后拷贝进来。

关键代码如下：

```python
model_hf = GPT2LMHeadModel.from_pretrained(model_type)
sd_hf = model_hf.state_dict()

transposed = [
    'attn.c_attn.weight',
    'attn.c_proj.weight',
    'mlp.c_fc.weight',
    'mlp.c_proj.weight'
]

for k in sd_keys_hf:
    if any(k.endswith(w) for w in transposed):
        sd[k].copy_(sd_hf[k].t())
    else:
        sd[k].copy_(sd_hf[k])
```

这让仓库兼顾了两类使用方式：

- 从头训练一个小模型。
- 直接站在 GPT-2 预训练权重上做评估或微调。

### 4.8 `configure_optimizers`：参数分组策略

优化器配置没有写在 `train.py` 中，而是放回模型内部。这样做的原因是“哪些参数该 decay”本质上属于模型结构知识。

对应代码如下：

```python
param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
```

判断规则非常直接：

- 2D 参数通常是矩阵权重，做 weight decay。
- 1D 参数通常是 bias / norm 权重，不做 weight decay。

### 4.9 `generate`：最小闭环的自回归采样

生成逻辑也完全放在模型里，保持调用端简单：

```python
idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
logits, _ = self(idx_cond)
logits = logits[:, -1, :] / temperature

if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')

probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
idx = torch.cat((idx, idx_next), dim=1)
```

这一实现虽然短，但已经具备三个非常关键的采样控制点：

- 上下文裁剪到 `block_size`
- `temperature`
- `top_k`

## 5. 训练模块 `train.py`

`train.py` 的设计核心，是把“训练一个 GPT 所需的工程能力”直接铺开在一个文件里，不做过度抽象。

### 5.1 默认参数 + 覆盖机制

脚本顶部直接写满默认超参数：

```python
out_dir = 'out'
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
learning_rate = 6e-4
max_iters = 600000
compile = True
```

然后立刻通过 `configurator.py` 覆盖：

```python
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
```

因此 `train.py` 既是脚本，也是“显式的配置说明书”。

### 5.2 DDP 初始化：尽量少样板代码

DDP 逻辑在这里没有做额外封装，而是直接依赖环境变量判断：

```python
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
```

同时它会把全局梯度累积步数按进程数等比缩小：

```python
assert gradient_accumulation_steps % ddp_world_size == 0
gradient_accumulation_steps //= ddp_world_size
```

这个处理很关键，因为它保证了总 token batch size 在 DDP 场景下仍然符合预期。

### 5.3 数据读取：面向连续 token 流的随机窗口采样

`get_batch` 设计得非常朴素，但和前面的二进制数据格式完美配套：

```python
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
```

它不是按“样本条目”采样，而是按“随机起点”切出长度为 `block_size` 的连续窗口：

- `x` 是输入序列
- `y` 是整体右移一位的监督目标

这正是自回归语言模型最自然的数据形式。

### 5.4 模型初始化：三种入口统一到同一训练脚本

模型初始化分成三种模式：

- `scratch`
- `resume`
- `gpt2*`

核心代码如下：

```python
if init_from == 'scratch':
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, override_args)
```

这段设计体现了一个重要原则：训练入口统一，初始化方式可切换。这样微调和继续训练都不需要另起一套脚本。

### 5.5 AMP、GradScaler 和 `torch.compile`

训练脚本默认启用现代 PyTorch 的几个关键优化：

```python
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if compile:
    model = torch.compile(model)
```

设计上它们被放在主脚本里，而不是封装进训练框架中，原因很直接：这些优化是“是否启用”的问题，不需要隐藏在深层抽象里。

### 5.6 评估与学习率调度

训练过程中每隔一段步数会执行验证：

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

学习率则采用“warmup + cosine decay”：

```python
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

这也是当前大模型训练里非常常见的一套调度策略。

### 5.7 主训练循环：把复杂度压缩在最关键的几步里

训练循环中的关键路径如下：

```python
for micro_step in range(gradient_accumulation_steps):
    if ddp:
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch('train')
    scaler.scale(loss).backward()

if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

这里面有几个很值得注意的设计点：

1. 用梯度累积模拟更大的全局 batch。
2. 只在最后一个 micro step 同步 DDP 梯度，减少通信开销。
3. 在反向传播前就预取下一批数据，尽量把数据准备与计算重叠。
4. 用 `set_to_none=True` 更快地清梯度。

### 5.8 检查点保存

保存逻辑也非常直接：

```python
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

这里把“模型参数 + 优化器状态 + 模型结构参数 + 训练步数 + 最优验证损失 + 原始配置”都打包进去，因此：

- 可以继续训练
- 可以推理
- 可以知道这个 checkpoint 是怎么训练出来的

## 6. 推理采样模块 `sample.py`

`sample.py` 的职责是把训练出的 checkpoint 重新拼成可运行模型，然后执行 prompt 编码和自回归生成。

### 6.1 模型恢复

如果是从训练结果恢复：

```python
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
```

如果直接使用 GPT-2：

```python
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```

因此推理脚本与训练脚本共享同一个模型定义，不存在两套模型代码。

### 6.2 编码器选择

这部分设计也很巧妙：如果 checkpoint 对应的数据集目录下有 `meta.pkl`，就优先使用训练时的编码方式；否则回退到 GPT-2 BPE。

```python
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
```

也就是说：

- 字符级模型会自动使用字符级词表。
- GPT-2 或 BPE 数据集默认使用 `tiktoken`。

### 6.3 生成流程

采样调用端保持得非常薄：

```python
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
```

这说明 `sample.py` 的设计理念是“把复杂逻辑留在模型内部，脚本只负责组装输入输出”。

## 7. 辅助模块 `bench.py`

`bench.py` 可以看成是训练路径的“去工程化版本”。它保留了这些核心步骤：

- 构造 batch
- 前向传播
- 反向传播
- 优化器更新
- MFU 统计

但拿掉了检查点、验证、DDP 管理等外围逻辑，所以很适合做：

- 性能 profiling
- `torch.compile` 对比
- 数据加载与计算瓶颈分析

## 8. 关键设计总结

nanoGPT 的关键模块之所以容易理解，是因为它在设计上刻意做了下面几件事：

1. 用极少的文件承载完整训练链路。
2. 让模型结构、训练逻辑和采样逻辑共享同一套核心实现。
3. 用 `memmap + 连续 token 流` 简化数据读取。
4. 用 `configurator.py` 让实验配置保持“脚本即配置”。
5. 在保持短小的同时，加入 DDP、AMP、Flash Attention、`torch.compile` 等实用优化。

如果从“学习源码”的角度看，这个仓库最值得参考的不是某一个函数，而是它的整体取舍：

- 不追求抽象层次最多
- 不追求框架感最强
- 追求可读、可改、可跑通

这也是为什么它非常适合作为 GPT 训练代码的阅读入口。

## 9. 建议的阅读顺序

如果你要顺着源码理解整个系统，推荐按下面顺序阅读：

1. `train.py`：先看训练脚本整体流程。
2. `model.py`：再看模型内部是如何完成一次前向传播的。
3. `sample.py`：理解 checkpoint 恢复和生成逻辑。
4. `configurator.py`：理解配置覆盖机制。
5. `data/shakespeare_char/prepare.py` 或 `data/openwebtext/prepare.py`：回看数据是如何被组织成 token 流的。

以上顺序最容易把“数据格式、模型结构、训练循环、推理流程”串成一个完整闭环。

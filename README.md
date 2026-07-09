<img width="100%" alt="babygrad" src="./header.png" />
A small neural network library built from scratch in pure Python. No NumPy. No PyTorch. No dependencies. Just for fun.

The project builds up incrementally from raw tensor operations, where each step depends on and extends what came before.

Designed to be a lightweight learning project that's easy to mess around with.

> [!CAUTION]
> Not intended for serious usage. Expect it to be slow - no C, no GPU.

## Rules

- **Zero external dependencies.** If it isn't in the Python standard library or written by hand, it doesn't belong here.

- **Every operation must be understood before it's implemented.** No copying reference code. No AI implementation.

A couple of caveats/exceptions: I have no desire to write my own plotting library, so matplotlib and graphviz are used for viz. There are also notebooks associated with each phase - those are AI generated and exist simply as a demo of the API and what was completed during that phase.

## Development

Install the project and development tools:

```bash
uv sync --dev
```

Run the test suite:

```bash
uv run pytest
```

Run notebooks with the `.venv` kernel. `uv sync --dev` installs `babygrad` as an editable package, so notebooks can import it without modifying `sys.path`.

### Visualization dependencies (optional)

Histograms and computation-graph diagrams use `matplotlib` and `graphviz`. The `graphviz` Python package is installed by `uv sync`, but graph rendering also needs the system `dot` binary:

```bash
# Debian/Ubuntu/WSL
sudo apt install graphviz
# macOS
brew install graphviz
```

Everything else runs without it.

## Roadmap

### Phase 1: Tensor Foundations

[Notebook demo](./notebooks/phase1_tensor_foundations.ipynb)

`*` = optional / nice-to-have — useful later, but not required to progress.

<details>
<summary><strong>Tensor data structure</strong> — storage, shape, indexing, and size metadata</summary>

- [x] Flat storage
- [x] Shape
- [x] Rank / dimension count
- [x] Element count
- [x] Basic indexing and offset calculation

</details>

<details>
<summary><strong>Element-wise operations</strong> — per-value unary and binary operations</summary>

- [x] Add
- [x] Subtract
- [x] Multiply
- [x] Divide
- [x] Negate
- [x] Exp
- [x] Log
- [x] Sqrt
- [x] Power

</details>

<details>
<summary><strong>Reduction operations</strong> — operations that collapse one or more axes</summary>

- [x] Sum
- [x] Max
- [x] Mean

</details>

<details open>
<summary><strong>Shape manipulation</strong> — changing how tensor data is arranged or viewed</summary>

- [x] Reshape
- [x] Transpose
- [x] Flatten
- [ ] Permute / swap axes\*
- [x] View vs copy semantics\* (copy is done)

</details>

<details open>
<summary><strong>Broadcasting</strong> — shape alignment and expansion rules</summary>

- [x] Scalar broadcasting
- [x] Singleton-dimension broadcasting
- [ ] Full NumPy-style broadcasting\*

</details>

<details open>
<summary><strong>Matrix multiplication</strong> — the core compute primitive</summary>

- [x] Vector dot product
- [x] Matrix-vector multiplication
- [x] Vector-matrix multiplication
- [x] Matrix-matrix multiplication
- [ ] Batched matrix multiplication\*

</details>

### Phase 2: Forward Neural Network Primitives

[Notebook demo](./notebooks/phase2_forward_nn_primitives.ipynb)

- [x] **Linear layer** — weights, biases, forward values
- [x] **Activation functions** — sigmoid, tanh, ReLU and softmax forward values
- [x] **Sequential model** — run ordered layers from input tensor to `y_pred`
- [x] **Loss functions** — MSE, cross-entropy loss values

### Phase 3: Autograd

[Notebook demo](./notebooks/phase3_autograd.ipynb)

<details>
<summary><strong>Computational graph</strong> — tracking operations as a DAG of nodes</summary>

- [x] add
- [x] sub
- [x] neg
- [x] mul
- [x] pow
- [x] max
- [x] sum
- [x] mean

- [x] matmul
- [x] transpose

- [x] exp
- [x] log
- [x] div

- [x] sigmoid, tanh, relu
- [x] softmax (by composition)

</details>

- [x] **Backward pass** — reverse-mode autodiff graph walk
- [x] **Gradient accumulation** — multi-use tensors propagate once after accumulating

### Phase 4: Training

[Notebook demo](./notebooks/phase4_optimization_training.ipynb)

- [x] **SGD optimizer** — parameter updates, learning rate
- [x] **Training loop** — forward, loss, backward, step
- [x] **Validation and test** - phases plus metrics
- [x] **Batching** - full, mini-batch training
- [x] **Zero grads between steps** — `.grad` is zeroed only at construction and deliveries use `+=`, so persistent tensors (weights) accumulate gradients across `backward()` calls unless reset each iteration
- [ ] **Momentum**\* — optimizer state carried across steps
- [ ] **L1 / L2 weight decay**\* — penalty-term regularization
- [ ] **Gradient clipping**\* — stabilize large updates
- [ ] **Early stopping**\* — halt on validation plateau

### Phase 5: Going Deeper

[Notebook demo](./notebooks/phase5_visualization.ipynb)

- [x] **Multi-layer perceptron** — stacking linear + activation layers
- [x] **Visualization** — plotting training history and graphing the network, the lens for what follows
- [x] **Vanishing gradients** — observing the problem firsthand with deep stacks
- [x] **Weight initialization** — Xavier/Glorot, He
  - [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Glorot & Bengio, 2010
  - [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — He et al., 2015
- [x] **Batch normalization**
  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy, 2015

### Phase 6: Data Pipeline

[Notebook demo](./notebooks/phase6_data_pipeline.ipynb)

- [x] **Dataset abstraction** — `Dataset` ABC with a per-source `load()` / `__getitem__`; `CSVDataset` loads tabular data eagerly into `Sample(features, target)` rows
- [x] **DataLoader** — shuffles a split and yields minibatches, with `full_batch()` for whole-split validation/test passes
- [x] **Per-batch tensorization** — collate raw rows into `(features, target)` tensors on the fly, replacing the up-front whole-dataset batching
- [x] **Fit-on-train preprocessing** — one-hot encoding fit on the training split and shared to val/test

### Phase 7: Residual Networks

[Notebook demo](./notebooks/phase7_residual_networks.ipynb)

- [x] **Skip connections** — the residual block as a solution to vanishing gradients
- [x] **Stacking residual blocks** — building a small ResNet
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015

### Phase 8: The Original Transformer

**Building blocks** — general techniques the transformer relies on:

- [x] **Dropout** — train/eval mode, inverted scaling (regularization once nets get deep)
- [x] **Layer normalization**
  - [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba et al., 2016
- [x] **Adam optimizer** — optimizer state, bias-corrected adaptive learning rates (the transformer's optimizer)
  - [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba, 2014
- [ ] **Learning-rate warmup schedule** — the Vaswani warmup (+ cosine decay\*)
  - cosine decay: [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Loshchilov & Hutter, 2016
- [ ] **Label smoothing**\* — soften one-hot targets
  - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) — Szegedy et al., 2016
- [ ] **Data pipeline — sequence collation** — pad + mask variable-length sequences and lazily load tokens by offset, extending the eager tabular collate

---

**The transformer** — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017):

- [ ] **Scaled dot-product attention** — queries, keys, values
- [ ] **Multi-head attention** — parallel attention heads, concatenation, projection
- [ ] **Sinusoidal positional encoding** — injecting sequence order
- [ ] **Position-wise feed-forward network** — the other half of a transformer block
- [ ] **Encoder and decoder blocks** — assembling the full architecture
- [ ] **Masking** — padding masks, causal (look-ahead) masks

### Phase 9: Modern Transformer Modifications

- [ ] **RMSNorm** — replacing LayerNorm, dropping the mean centering
  - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019
- [ ] **SwiGLU** — gated linear units replacing ReLU in the FFN
  - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [ ] **Rotary Position Embedding (RoPE)** — rotation-based positional encoding replacing sinusoidal
  - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021

### Phase 10: Efficient Attention

- [ ] **Multi-Query Attention (MQA)** — single shared KV head across all query heads
  - [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer, 2019
- [ ] **Grouped-Query Attention (GQA)** — intermediate KV head sharing
  - [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [ ] **Sliding window attention** — fixed-size local attention windows
  - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy et al., 2020
  - [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023

### Phase 11: Inference Optimizations

- [ ] **KV-cache** — caching key/value pairs for autoregressive generation
- [ ] **Speculative decoding** — draft model + verification for parallel token generation
  - [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022
  - [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al., 2023

### Phase 12: Mixture of Experts

- [ ] **Sparse gating** — routing tokens to a subset of expert FFNs
- [ ] **MoE transformer block** — integrating sparse experts into the transformer
  - [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017

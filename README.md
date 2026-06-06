<img width="100%" alt="babygrad" src="./header.png" />
A small neural network library built from scratch in pure Python. No NumPy. No PyTorch. No dependencies. Just for fun.

The project builds up incrementally from raw tensor operations, where each step depends on and extends what came before.

Designed to be a lightweight learning project that's easy to mess around with.

> [!CAUTION]
> Not intended for serious usage. Expect it to be slow - no C, no GPU.

## Rules

- **Zero external dependencies.** If it isn't in the Python standard library or written by hand, it doesn't belong here (a few minor exceptions for things like matplotlib)

- **Every operation must be understood before it's implemented.** No copying reference code. No AI implementation.

## Roadmap

### Phase 1: Tensor Foundations

`*` = useful later, but not required before building forward-only layers.

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

- [x] **Linear layer** — weights, biases, forward values
- [x] **Activation functions** — ReLU and softmax forward values
- [x] **Sequential model** — run ordered layers from input tensor to `y_pred`
- [x] **Loss functions** — MSE, cross-entropy loss values

### Phase 3: Autograd

<details open>
<summary><strong>Computational graph</strong> — tracking operations as a DAG of nodes</summary>

- [x] add
- [ ] sub
- [ ] neg
- [ ] mul
- [ ] pow
- [ ] sum
- [ ] mean
- [ ] matmul
- [ ] transpose
- [ ] exp
- [ ] log
- [ ] div
- [ ] relu
- [ ] max
- [ ] softmax

</details>

- [ ] **Backward pass** — reverse-mode autodiff, chain rule
- [ ] **Gradient accumulation** — handling multi-use tensors

### Phase 4: Training

- [ ] **SGD optimizer** — parameter updates, learning rate
- [ ] **Training loop** — forward, loss, backward, step

### Phase 5: Going Deeper

- [x] **Multi-layer perceptron** — stacking linear + activation layers
- [ ] **Vanishing gradients** — observing the problem firsthand with deep stacks
- [ ] **Weight initialization** — Xavier/Glorot, He
  - [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Glorot & Bengio, 2010
  - [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — He et al., 2015
- [ ] **Batch normalization**
  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy, 2015

### Phase 6: Residual Networks

- [ ] **Skip connections** — the residual block as a solution to vanishing gradients
- [ ] **Stacking residual blocks** — building a small ResNet
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015

### Phase 7: The Original Transformer

- [ ] **Scaled dot-product attention** — queries, keys, values
- [ ] **Multi-head attention** — parallel attention heads, concatenation, projection
- [ ] **Sinusoidal positional encoding** — injecting sequence order
- [ ] **Position-wise feed-forward network** — the other half of a transformer block
- [ ] **Layer normalization**
  - [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba et al., 2016
- [ ] **Encoder and decoder blocks** — assembling the full architecture
- [ ] **Masking** — padding masks, causal (look-ahead) masks
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017

### Phase 8: Modern Transformer Modifications

- [ ] **RMSNorm** — replacing LayerNorm, dropping the mean centering
  - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019
- [ ] **SwiGLU** — gated linear units replacing ReLU in the FFN
  - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [ ] **Rotary Position Embedding (RoPE)** — rotation-based positional encoding replacing sinusoidal
  - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021

### Phase 9: Efficient Attention

- [ ] **Multi-Query Attention (MQA)** — single shared KV head across all query heads
  - [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer, 2019
- [ ] **Grouped-Query Attention (GQA)** — intermediate KV head sharing
  - [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [ ] **Sliding window attention** — fixed-size local attention windows
  - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy et al., 2020
  - [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023

### Phase 10: Inference Optimizations

- [ ] **KV-cache** — caching key/value pairs for autoregressive generation
- [ ] **Speculative decoding** — draft model + verification for parallel token generation
  - [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022
  - [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al., 2023

### Phase 11: Mixture of Experts

- [ ] **Sparse gating** — routing tokens to a subset of expert FFNs
- [ ] **MoE transformer block** — integrating sparse experts into the transformer
  - [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017

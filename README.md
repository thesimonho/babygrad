# babygrad

A neural network library built from scratch in pure Python. No NumPy. No PyTorch. No dependencies.

The project builds up incrementally from raw tensor operations to a modern transformer, where each step depends on and extends what came before.

## Rules

- **Zero external dependencies.** If it isn't in the Python standard library or written by hand, it doesn't belong here.
- **Every operation must be understood before it's implemented.** No copying reference code.

## Roadmap

### Phase 1: Tensor Foundations

`*` = useful later, but not required before starting autograd.

- [x] **Tensor data structure** — storage, shape, indexing, and size metadata
  - [x] Flat storage
  - [x] Shape
  - [x] Rank / dimension count\*
  - [x] Element count\*
  - [x] Basic indexing and offset calculation
- [x] **Element-wise operations** — per-value unary and binary operations
  - [x] Add
  - [x] Subtract
  - [x] Multiply
  - [x] Divide
  - [ ] Negate\*
  - [ ] Exp\*
  - [ ] Log\*
  - [ ] Sqrt\*
  - [ ] Power\* (optional unless later math needs it directly)
- [ ] **Reduction operations** — operations that collapse one or more axes
  - [ ] Sum
  - [ ] Max\*
  - [ ] Mean\*
- [ ] **Shape manipulation** — changing how tensor data is arranged or viewed
  - [ ] Reshape
  - [x] Transpose
  - [ ] Flatten\*
  - [ ] Permute / swap axes\*
  - [ ] View vs copy semantics\* (optional at first)
- [ ] **Broadcasting** — shape alignment and expansion rules
  - [ ] Scalar broadcasting
  - [ ] Singleton-dimension broadcasting\*
  - [ ] Full NumPy-style broadcasting\* (optional at first)
- [x] **Matrix multiplication** — the core compute primitive
  - [x] Vector dot product
  - [x] Matrix-vector multiplication
  - [x] Vector-matrix multiplication
  - [x] Matrix-matrix multiplication
  - [ ] Batched matrix multiplication\*

### Phase 2: Autograd

- [ ] **Computational graph** — tracking operations as a DAG of nodes
- [ ] **Backward pass** — reverse-mode autodiff, chain rule
- [ ] **Gradient accumulation** — handling multi-use tensors

### Phase 3: Neural Network Primitives

- [ ] **Linear layer** — weights, biases, forward and backward
- [ ] **Activation functions** — sigmoid, tanh, ReLU
- [ ] **Loss functions** — MSE, cross-entropy
- [ ] **SGD optimizer** — parameter updates, learning rate
- [ ] **Training loop** — forward, loss, backward, step

### Phase 4: Going Deeper

- [ ] **Multi-layer perceptron** — stacking linear + activation layers
- [ ] **Vanishing gradients** — observing the problem firsthand with deep stacks
- [ ] **Weight initialization** — Xavier/Glorot, He
  - [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Glorot & Bengio, 2010
  - [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — He et al., 2015
- [ ] **Batch normalization**
  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy, 2015

### Phase 5: Residual Networks

- [ ] **Skip connections** — the residual block as a solution to vanishing gradients
- [ ] **Stacking residual blocks** — building a small ResNet
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015

### Phase 6: The Original Transformer

- [ ] **Scaled dot-product attention** — queries, keys, values
- [ ] **Multi-head attention** — parallel attention heads, concatenation, projection
- [ ] **Sinusoidal positional encoding** — injecting sequence order
- [ ] **Position-wise feed-forward network** — the other half of a transformer block
- [ ] **Layer normalization**
  - [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba et al., 2016
- [ ] **Encoder and decoder blocks** — assembling the full architecture
- [ ] **Masking** — padding masks, causal (look-ahead) masks
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017

### Phase 7: Modern Transformer Modifications

- [ ] **RMSNorm** — replacing LayerNorm, dropping the mean centering
  - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019
- [ ] **SwiGLU** — gated linear units replacing ReLU in the FFN
  - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [ ] **Rotary Position Embedding (RoPE)** — rotation-based positional encoding replacing sinusoidal
  - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021

### Phase 8: Efficient Attention

- [ ] **Multi-Query Attention (MQA)** — single shared KV head across all query heads
  - [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer, 2019
- [ ] **Grouped-Query Attention (GQA)** — intermediate KV head sharing
  - [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [ ] **Sliding window attention** — fixed-size local attention windows
  - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy et al., 2020
  - [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023

### Phase 9: Inference Optimizations

- [ ] **KV-cache** — caching key/value pairs for autoregressive generation
- [ ] **Speculative decoding** — draft model + verification for parallel token generation
  - [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022
  - [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al., 2023

### Phase 10: Mixture of Experts

- [ ] **Sparse gating** — routing tokens to a subset of expert FFNs
- [ ] **MoE transformer block** — integrating sparse experts into the transformer
  - [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017

## Status

Just getting started.

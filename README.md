# babygrad

A neural network library built from scratch in pure Python. No NumPy. No PyTorch. No dependencies.

The project builds up incrementally from raw tensor operations to a modern transformer, where each step depends on and extends what came before.

## Rules

- **Zero external dependencies.** If it isn't in the Python standard library or written by hand, it doesn't belong here.
- **Every operation must be understood before it's implemented.** No copying reference code.

## Roadmap

### Phase 1: Tensor Foundations

1. **Tensor data structure** — storage, shape, strides, indexing
2. **Element-wise operations** — add, subtract, multiply, divide, power
3. **Reduction operations** — sum, mean, max
4. **Shape manipulation** — reshape, transpose, views vs copies
5. **Broadcasting** — shape alignment and expansion rules
6. **Matrix multiplication** — the core compute primitive

### Phase 2: Autograd

1. **Computational graph** — tracking operations as a DAG of nodes
2. **Backward pass** — reverse-mode autodiff, chain rule
3. **Gradient accumulation** — handling multi-use tensors

### Phase 3: Neural Network Primitives

1. **Linear layer** — weights, biases, forward and backward
2. **Activation functions** — sigmoid, tanh, ReLU
3. **Loss functions** — MSE, cross-entropy
4. **SGD optimizer** — parameter updates, learning rate
5. **Training loop** — forward, loss, backward, step

### Phase 4: Going Deeper

1. **Multi-layer perceptron** — stacking linear + activation layers
2. **Vanishing gradients** — observing the problem firsthand with deep stacks
3. **Weight initialization** — Xavier/Glorot, He
    - [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Glorot & Bengio, 2010
    - [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — He et al., 2015
4. **Batch normalization**
    - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy, 2015

### Phase 5: Residual Networks

1. **Skip connections** — the residual block as a solution to vanishing gradients
2. **Stacking residual blocks** — building a small ResNet
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015

### Phase 6: The Original Transformer

1. **Scaled dot-product attention** — queries, keys, values
2. **Multi-head attention** — parallel attention heads, concatenation, projection
3. **Sinusoidal positional encoding** — injecting sequence order
4. **Position-wise feed-forward network** — the other half of a transformer block
5. **Layer normalization**
    - [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba et al., 2016
6. **Encoder and decoder blocks** — assembling the full architecture
7. **Masking** — padding masks, causal (look-ahead) masks
    - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017

### Phase 7: Modern Transformer Modifications

1. **RMSNorm** — replacing LayerNorm, dropping the mean centering
    - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019
2. **SwiGLU** — gated linear units replacing ReLU in the FFN
    - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
3. **Rotary Position Embedding (RoPE)** — rotation-based positional encoding replacing sinusoidal
    - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021

### Phase 8: Efficient Attention

1. **Multi-Query Attention (MQA)** — single shared KV head across all query heads
    - [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer, 2019
2. **Grouped-Query Attention (GQA)** — intermediate KV head sharing
    - [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
3. **Sliding window attention** — fixed-size local attention windows
    - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy et al., 2020
    - [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023

### Phase 9: Inference Optimizations

1. **KV-cache** — caching key/value pairs for autoregressive generation
2. **Speculative decoding** — draft model + verification for parallel token generation
    - [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022
    - [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al., 2023

### Phase 10: Mixture of Experts

1. **Sparse gating** — routing tokens to a subset of expert FFNs
2. **MoE transformer block** — integrating sparse experts into the transformer
    - [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017

## Status

Just getting started.

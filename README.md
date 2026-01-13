# Axon

[![CI](https://github.com/abidanBrito/axon/actions/workflows/ci.yml/badge.svg)](https://github.com/abidanBrito/axon/actions/workflows/ci.yml)
[![STATUS](https://img.shields.io/badge/status-WIP-red.svg)](https://github.com/abidanBrito/axon)

A lightweight deep learning framework built from first principles in modern C++.

## Features

### Architecture
- [x] Feedforward networks with arbitrary topology.
- [x] Bias neurons for learning offsets.
- [ ] Convolutional layers.
- [ ] Max/average pooling (downscaling).
- [ ] Transpose convolutions (upscaling).
- [ ] Batch normalization.
- [ ] Layer normalization.
- [ ] Sequential model API.

### Training
- [x] Backpropagation with automatic gradient computation.
- [x] Gradient descent with momentum optimizer.
- [x] Configurable learning rate and momentum.
- [ ] Mini-batch training.
- [ ] Exponential moving average.
- [ ] L2 regularization (weight decay).
- [ ] Dropout.
- [ ] Gradient clipping.
- [ ] Data loaders with shuffling.
- [ ] Model serialization (save/load weights).
- [ ] Model checkpointing (save/resume training).
- [ ] Weight initialization strategies: Xavier/Glorot, He/Kaiming.
- [ ] Advanced optimizers: Adam, AdamW, RMSprop.
- [ ] Learning rate schedulers: step decay, reduce on plateau, cosine annealing.
- [ ] Data augmentation (random flips, crops, rotations).
- [ ] Mixed precision training (FP16/FP32).

### Activation functions
- [x] Linear (identity).
- [x] Hyperbolic tangent.
- [x] ReLU (Rectified Linear Unit).
- [x] Sigmoid.
- [ ] Softmax.
- [ ] ELU (Exponential Linear Unit).
- [ ] GELU (Gaussian Error Linear Unit).
- [ ] Swish/SiLU.

### Loss functions
- [x] Mean Squared Error (MSE).
- [ ] Mean Absolute Error (MAE).
- [ ] Binary Cross-Entropy (BCE).
- [ ] Categorical Cross-Entropy.
- [ ] Dice Loss.
- [ ] Combo Loss.
- [ ] IoU/Jaccard Loss.
- [ ] Focal Loss.
- [ ] Tversky Loss.
- [ ] Huber Loss.

### Performance
- [ ] Memory pool allocators.
- [ ] Multi-threaded batch processing.
- [ ] SIMD intrinsics (AVX2/AVX-512)
- [ ] CUDA support (GPU acceleration).
- [ ] cuDNN integration (optimized conv/pooling).

### Utilities
- [ ] Built-in metrics: accuracy, precision, recall, F1.
- [ ] Model summary (print architecture).
- [ ] Training history logging.
- [ ] Visualization tools.
- [ ] Confusion matrix.
- [ ] Youden's J index.

## LICENSE
This repository is released under the MIT license. See [LICENSE](LICENSE) for more information.

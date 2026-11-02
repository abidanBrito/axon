# Axon

[![CI](https://github.com/abidanBrito/axon/actions/workflows/ci.yml/badge.svg)](https://github.com/abidanBrito/axon/actions/workflows/ci.yml)
[![STATUS](https://img.shields.io/badge/status-WIP-red.svg)](https://github.com/abidanBrito/axon)

A lightweight deep learning framework built from first principles in modern C++.

## Features

### Core
- [ ] Tensor abstraction (contiguous N-dimensional storage).
- [ ] Matrix-based dense (fully-connected) layers: weight matrix + bias vector.
- [ ] Refactor per-neuron object model to vectorized layers.

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
- [ ] Autograd engine (dynamic computational graph).
- [x] Gradient descent with momentum optimizer.
- [x] Configurable learning rate and momentum.
- [ ] Mini-batch training.
- [ ] Exponential moving average.
- [ ] L2 regularization (weight decay).
- [ ] Dropout.
- [ ] Gradient clipping.
- [ ] Data loaders with shuffling.
- [ ] Seedable RNG for reproducible runs.
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

### Testing
- [ ] Gradient checking (finite-difference verification).
- [ ] Overfit a single batch to near-zero loss (Karpathy, 2019).

## References
- David E. Rumelhart et al., [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0) (1986).
- George Cybenko, [Approximation by superpositions of a sigmoidal function](https://doi.org/10.1007/BF02551274) (1989).
- Yann LeCun et al., [Efficient BackProp](https://doi.org/10.1007/3-540-49430-8_2) (1998).
- David Miller, [Neural Net in C++ Tutorial](https://millermattson.com/dave/?p=54).
- Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) (2015).
- John D. Kelleher, [Deep Learning](https://mitpress.mit.edu/9780262537551/deep-learning/) (2019).
- Andrej Karpathy, [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (2019).

## LICENSE
This repository is released under the MIT license. See [LICENSE](LICENSE) for more information.

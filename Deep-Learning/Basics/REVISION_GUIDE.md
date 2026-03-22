# 📋 Deep Learning Basics: Perceptrons Revision Guide

The Perceptron is the building block of deep neural networks. It is the simplest type of artificial neural network.

---

## ⚡ Algorithms in this Folder

1. **Simple Perceptron**: 
   - A single-layer neural network with binary-step activation.
   - Takes inputs, multiplies them by weights, adds a bias, and passes it through an activation function.
   - Equation: $f(x) = \sum(w_i x_i) + b$
2. **Perceptron from Scratch**:
   - Manually calculating weight updates using the Error term.
   - Using bitwise operations (AND, OR, XOR) as classic learning examples.
3. **Training Iteration**: 
   - **Predict** $\rightarrow$ **Calculate Error** $\rightarrow$ **Adjust Weights** $\rightarrow$ **Converge**.

---

## 🛠️ Flow Structure

![Neural Network Workflow](../../assets/neural_network_workflow.png)

### Limitations:
- Single-layer perceptrons can only solve **linearly separable** problems.
- For non-linear problems (like XOR gate), we need **Multi-Layer Perceptrons (MLP)**.

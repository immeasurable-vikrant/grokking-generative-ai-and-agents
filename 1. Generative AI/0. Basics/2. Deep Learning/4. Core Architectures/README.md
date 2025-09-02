# 🧠 Deep Learning - Section D: Core Architectures

---

## 23. Feedforward Neural Networks (FNN)

Also known as **Multi-Layer Perceptrons (MLPs)**, this is the most basic type of neural network.

### 🧠 First Principles:
- Data flows in **one direction**: Input → Hidden Layers → Output
- No loops or memory; purely functional mapping
- Each layer transforms its input using **weights + activation functions**

### Use Cases:
- Tabular data (e.g., predicting price, classifying users)
- Works well when **relationships are static and non-sequential**

💡 FNNs build the **core structure** used in other DL models like CNNs and Transformers.

---

## 24. Convolutional Neural Networks (CNN)

CNNs are designed to handle **spatially structured data** like images, video frames, or even grids.

### 🧠 First Principles:
- Use **convolutional layers** to detect patterns (edges, textures) by sliding filters over the data
- Capture **local relationships** while reducing the number of parameters (compared to FNNs)
- Followed by **pooling layers** that downsample and reduce dimensions

### Why Needed?
- FNNs don’t scale well to image data (too many parameters)
- CNNs exploit **spatial locality and weight sharing** to efficiently process visual input

### Use Cases:
- Image classification (e.g., cats vs dogs)
- Object detection (e.g., self-driving cars)
- Medical imaging, face recognition

💡 Early GenAI systems like DALL·E use CNNs in their **encoder/decoder pipelines**.

---

## 25. Recurrent Neural Networks (RNN)

RNNs are built for **sequential data** — where past information influences future predictions.

### 🧠 First Principles:
- Has a **loop** that allows info to persist from one time step to the next
- At each time step, it receives an input and its own previous output
- The same weights are reused across all time steps

### Why Needed?
- FNNs and CNNs don’t remember past inputs — they treat each input as independent
- RNNs introduce **memory**, enabling models to learn **temporal dependencies**

### Use Cases:
- Text generation, speech recognition, time-series forecasting
- Language modeling (pre-Transformer era)

### Limitations:
- Struggles with long-term memory (vanishing gradients)
- Training can be slow and unstable

💡 RNNs are the **stepping stones** to more advanced sequence models like LSTMs and Transformers.

---

## 26. Autoencoders

Autoencoders are **unsupervised neural networks** used for **compression and reconstruction**.

### 🧠 First Principles:
- Composed of two parts:
  - **Encoder**: Compresses input into a low-dimensional vector (latent space)
  - **Decoder**: Reconstructs original input from compressed form
- Trained to minimize **reconstruction loss** between input and output

### Use Cases:
- Dimensionality reduction (like PCA)
- Denoising data (e.g., blurry images)
- Pretraining for downstream tasks

💡 Important conceptual basis for **generative models** (like Variational Autoencoders, diffusion models, etc.)

---

## 27. When to Use What?

| Architecture | Best For | Key Strength |
|--------------|----------|--------------|
| **FNN** | Structured/tabular data | Simplicity and generalization |
| **CNN** | Image/video/grid data | Spatial feature extraction |
| **RNN** | Sequential/time-series data | Temporal memory |
| **Autoencoders** | Compression/anomaly detection | Learning latent representations |

🧠 These architectures reflect how DL adapts to different data types — a key principle before you move to Transformers, which generalize these ideas further.

---

# ✅ Summary

- FNNs are the foundation — useful for general prediction tasks.
- CNNs specialize in **spatial understanding** of visual data.
- RNNs specialize in **temporal understanding** of sequential data.
- Autoencoders help in **reconstruction, denoising**, and dimensionality reduction.
- These architectures evolved based on the **nature of data** — image, sequence, structured, etc.

---

🧭 Next: **Section E – Practical Concepts in Deep Learning**
(Transfer learning, GPUs, frameworks, fine-tuning)

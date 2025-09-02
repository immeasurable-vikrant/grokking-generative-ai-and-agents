
📌 Think of it as a **math function that makes a decision** — like “Is this email spam?”

---

## 7. Weights, Biases, Layers

- **Weights** determine the importance of each input.
- **Bias** allows the model to shift the decision boundary.
- **Layers**:
  - **Input layer**: Raw data
  - **Hidden layers**: Where learning happens
  - **Output layer**: Prediction or decision

🧠 More layers → More abstraction → More powerful representations  
💡 In GenAI, models like GPT-4 have **hundreds of layers**!

---

## 8. Activation Functions

These introduce **non-linearity** into the network so it can learn complex patterns.

| Function | Purpose | Range | Used In |
|----------|---------|-------|---------|
| Sigmoid  | Smooth binary output | 0 to 1 | Early networks |
| Tanh     | Centered activation | -1 to 1 | RNNs |
| ReLU     | Fast & sparse learning | 0 to ∞ | Most deep nets |
| Softmax  | Converts to probabilities | 0 to 1 | Classification output |

🧠 Without activation functions, the network becomes a **linear model** no matter how many layers it has.

---

## 9. Forward Propagation

This is how data **flows through the network** during prediction.

### Steps:
1. Multiply inputs by weights
2. Add bias
3. Pass result through activation
4. Send to next layer
5. Repeat until output

💡 In simple terms: It’s like **stacking math functions** to process the input.

---

## 10. Loss Functions

The **loss** tells the model **how wrong its prediction is**.

### Common Losses:
| Type | Function | Use Case |
|------|----------|----------|
| Regression | Mean Squared Error (MSE) | Predicting numbers |
| Classification | Cross-Entropy Loss | Classifying categories |

🧠 The goal of training is to **minimize the loss** — smaller loss means better predictions.

---

## 11. Gradient Descent

Gradient Descent is the **optimization algorithm** used to minimize the loss.

### 🧠 First Principles:
- Computes the gradient (slope) of the loss function w.r.t. weights.
- Takes small steps in the direction that **reduces error**.
- Think of it like **rolling downhill to the bottom of a valley** (where loss is lowest).

### Key term:
- **Learning Rate**: Controls step size — too small = slow, too big = unstable.

---

## 12. Backpropagation

Backpropagation is the **learning algorithm** of neural networks.

### How it works:
1. Perform **forward pass** to compute loss.
2. Use **chain rule of calculus** to compute gradients.
3. Propagate error **backwards** through layers.
4. Update weights using **gradient descent**.

💡 It's like teaching the network by saying:  
> “Here’s what you predicted. Here’s how wrong you were. Adjust your weights accordingly.”

---

## 13. Learning Rate, Epochs, Batch Size

- **Learning Rate**: Controls how much to update weights during training.
- **Epoch**: One full pass through the entire training dataset.
- **Batch Size**: Number of samples the model processes before updating weights.

🧠 Tuning these values impacts both **speed and stability** of learning.

---

## 14. Overfitting vs Underfitting

| Term | Meaning | Symptom | Cause |
|------|---------|---------|-------|
| Overfitting | Model memorizes training data | High train accuracy, low test accuracy | Too complex |
| Underfitting | Model fails to learn | Low accuracy everywhere | Too simple or undertrained |

💡 Overfitting = Too smart  
💡 Underfitting = Too dumb  
✅ The goal is **generalization**: perform well on **unseen data**.

---

## 15. Regularization Techniques (Dropout, L1/L2)

Regularization prevents overfitting by **penalizing complexity**.

| Method | What It Does |
|--------|---------------|
| **L1 (Lasso)** | Encourages sparse weights (some = 0) |
| **L2 (Ridge)** | Shrinks all weights slightly |
| **Dropout** | Randomly drops neurons during training to prevent dependency |

💡 Regularization makes the model **simpler**, more robust, and less likely to memorize.

---

## 16. Model Evaluation Metrics

Beyond loss, we use metrics to evaluate real-world performance.

| Metric | Use Case |
|--------|----------|
| **Accuracy** | Overall correctness |
| **Precision** | % of predicted positives that were correct |
| **Recall** | % of actual positives that were correctly predicted |
| **F1 Score** | Harmonic mean of precision & recall |
| **Confusion Matrix** | Detailed error breakdown |

🧠 Choose metrics that match your **business goal** — not always just accuracy.

---

# ✅ Summary

- Neural networks are built from perceptrons, layers, and activation functions.
- They learn via **forward pass → loss → backpropagation → weight update**.
- Concepts like overfitting, regularization, and metrics ensure models don’t just memorize but **generalize**.
- These fundamentals are the **core mechanics behind all deep learning models**, including GPT, DALL·E, and more.

---

🧭 Next: Section C – Training Deep Neural Networks Efficiently  
(Weight init, optimizers, batch norm, etc.)

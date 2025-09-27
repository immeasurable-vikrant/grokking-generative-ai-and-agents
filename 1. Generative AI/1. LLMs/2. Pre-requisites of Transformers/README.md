# 📚 Prerequisites for Understanding Transformers

This guide covers foundational concepts necessary before diving into the Transformer architecture, focusing on RNNs, LSTMs, and attention mechanisms—all explained in a developer-friendly, intuitive way.

---

## 1. Recurrent Neural Networks (RNNs)

### 🔧 What is an RNN?

An RNN (Recurrent Neural Network) is a type of neural network designed to handle sequential data such as text, audio, or time series. It processes one element at a time in a sequence, maintaining a "memory" (called hidden state) of previous inputs.

### 🧠 How it works:

At each time step \( t \), it:  
- Takes the input \( x_t \)  
- Combines it with the previous hidden state \( h_{t-1} \)  
- Produces a new hidden state \( h_t \) using a function (often a tanh activation)  

This allows it to "remember" past context.  

**Example:**  
Processing the sentence "The cat sat on the mat."  
- At time step 1: Input = "The" → produces hidden state \( h_1 \)  
- At time step 2: Input = "cat" + \( h_1 \) → produces \( h_2 \)  
- ... and so on.  
Each word depends on the memory of the previous ones.

### 🧨 Why RNNs Fail (or Struggle):

- **Vanishing/Exploding Gradients:** As sequences get longer, gradients during training either vanish (go to zero) or explode (go to infinity), making learning difficult.  
- **Long-Term Dependencies:** RNNs forget what happened long ago; important information from earlier steps often disappears.  
- **Slow Computation:** They can't be parallelized easily since each step depends on the output of the previous one.

---

## 2. LSTMs and GRUs (RNN Upgrades)

### 🔁 LSTM (Long Short-Term Memory)

LSTM improves on RNNs by introducing **gates** that control the flow of information:

- **Input Gate:** Decides what new information to store.  
- **Forget Gate:** Decides what information to discard.  
- **Output Gate:** Decides what information to output.  

### ⚙️ How it works:

- Maintains a **cell state** alongside the hidden state.  
- Uses gates to selectively add or remove information from the cell state.  

### ✅ Improvements over RNN:

- Better at capturing long-term dependencies.  
- Less prone to vanishing gradients.  

### ❌ Still problematic:

- Sequential processing remains (still slow).  
- More complex with many parameters, making training harder.  
- Can still struggle with very long sequences.

---

## 3. The Real Problem: Sequential Bottleneck

Both RNNs and LSTMs process data **sequentially**:  
- Must process one token at a time, in order.  
- No parallelization → slow training.  
- Long-range dependencies remain a challenge.

---

## 🔎 Enter Attention (The Big Idea)

Before transformers, a key innovation was introduced in 2014:  
### "Attention Is All You Need to Look Back"  
[Bahdanau et al., 2014]

### What is Attention?

A mechanism allowing a model to "look back" at **all previous inputs** and decide what to focus on, instead of relying only on the last hidden state.

### Real-world Analogy:

When reading a sentence, you don’t remember every single word. You focus on the important words when interpreting the meaning of the current word. This selective focusing is attention.

---

## 👁️‍🗨️ Attention Mechanism: How it Works

Given:  
- **Query (Q):** What you're looking for.  
- **Keys (K):** What each word "offers."  
- **Values (V):** What each word "contains."  

Attention computes a weighted sum:  
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

The weights are learned during training, allowing the model to focus on the most relevant parts of input.

---

## 🧠 Self-Attention (a Special Type)

Self-Attention is attention **within a sequence itself**:  
- For every word, compare it against all other words.  
- Decide how much each word contributes to the final representation.  

This connects distant words easily, e.g.,  
*“The cat that the dog chased was black.”*  
"Cat" and "was" are far apart, but self-attention can link them effectively.

---

## 🧨 Why Self-Attention Solves RNN Problems

| Feature                   | RNN/LSTMs | Self-Attention         |
|---------------------------|-----------|-----------------------|
| Sequential processing     | ❌ Yes    | ✅ No                 |
| Parallelizable            | ❌ No     | ✅ Yes                |
| Long-range dependencies    | ❌ Hard   | ✅ Easy               |
| Memory of entire sequence | ❌ Limited| ✅ Full context       |
| Training speed             | ❌ Slow   | ✅ Fast               |

---

## 🛠️ Summary of Prerequisites Before Transformers

| Concept       | Key Idea                      | Weakness                          |
|---------------|-------------------------------|----------------------------------|
| **RNN**        | Hidden state encodes past     | Struggles with long dependencies |
| **LSTM**       | Uses gates for selective memory| Still sequential, complex        |
| **Attention**  | Focus on relevant parts       | Computational cost can be high   |
| **Self-Attention** | Each word attends to all tokens | Computation over all token pairs |

---

## 🔜 What's Next?

Now that you understand the problems with RNNs/LSTMs and the intuition behind attention/self-attention, you're ready to dive into:  
- The **Transformer architecture**.  
- Components like **multi-head attention**, positional encoding, encoder-decoder structure.  
- How transformers are used in real-world models (e.g., GPT, BERT).

---
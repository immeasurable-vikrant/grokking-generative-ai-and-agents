# LLM Basics & Evolution

> A gentle but deep walkthrough from simple recurrent nets to modern large language models (RNN → LSTM → Seq2Seq → Attention → Transformers → GPT / BERT).

**Audience:** beginners with some programming/math background who want intuition, mechanics, timeline (when), and small examples.

---

## Table of Contents

1. TL;DR (quick map)
2. Timeline — when things happened and why they mattered
3. Recurrent Neural Networks (RNNs)

   * What
   * Why (the motivation)
   * How (equations & simple example)
   * Limitations
4. LSTM (Long Short-Term Memory)

   * What
   * Why it fixes RNN problems
   * How (gates and equations, step-by-step intuition)
5. Sequence-to-Sequence (Seq2Seq)

   * What (encoder-decoder)
   * Why (use case: translation)
   * How (teacher forcing, decoding, beam search)
6. Attention

   * Intuition
   * Bahdanau (additive) attention vs Scaled Dot-Product
   * How attention is computed (formulas & intuition)
7. Transformers

   * Why attention-only? (motivation)
   * The transformer block (self-attention, multi-head, FFN, residuals, layer norm)
   * Positional encoding
   * How training & inference differ
8. GPT vs BERT (and model families)

   * Architectures (decoder-only vs encoder-only vs encoder-decoder)
   * Training objectives (causal vs masked)
   * Typical use-cases
9. Practical details & next steps

   * Tokenization, vocabulary, batching, loss, evaluation
   * Limitations and common fixes (long context, efficiency)
10. Short code snippets (pseudocode / PyTorch-like for intuition)
11. Glossary & references

---

## 1) TL;DR (quick map)

* **RNNs**: neural nets with recurrence that process sequences one step at a time. Good for short dependencies.
* **LSTM**: RNN variant with gates to remember / forget — handles longer dependencies.
* **Seq2Seq**: encoder-decoder pattern that maps input sequences to output sequences (e.g., translation).
* **Attention**: lets the decoder peek at *which parts of the input* matter at every decoding step.
* **Transformers**: rely on attention (no recurrence), enabling massive parallelism and scale.
* **GPT (decoder-only)**: trained to predict next token (great at generation).
* **BERT (encoder-only)**: trained to fill masked tokens and produce contextual embeddings (great at understanding/classification).

Each new step answers shortcomings of the previous step: RNNs → LSTM fixes vanishing gradients → Seq2Seq adds encoder/decoder for mapping sequences → Attention fixes fixed-size bottleneck → Transformers drop recurrence and scale efficiently.

---

## 2) Timeline — When (high-level)

* 1980s–1990s: Recurrent networks and simple RNNs developed and experimented with (Elman networks, Jordan networks).
* 1997: LSTM introduced (Hochreiter & Schmidhuber) — solved long-term dependency issues.
* 2014: Sequence-to-Sequence with neural networks becomes popular for translation (Sutskever et al.).
* 2014: Bahdanau et al. introduced attention mechanism for NMT (align & translate).
* 2017: *Attention is All You Need* — Transformers introduced (Vaswani et al.).
* 2018–2019: BERT (Devlin et al.) and GPT (Radford et al.) popularized pretraining + fine-tuning and generative pretraining. From here, model scaling and LLM era takes off.

(These dates are the canonical moments — research continued to refine, extend, and scale these ideas.)

---

## 3) Recurrent Neural Networks (RNNs)

### What

RNNs process a sequence (x_1, x_2, \dots, x_T) one step at a time and maintain a hidden state (h_t) that summarizes the past.

### Why

Many tasks are sequential: text, audio, time-series. RNNs let information from earlier tokens influence later predictions.

### How (core equation)

A simple RNN step:

[ h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h) ]
[ y_t = W_{hy} h_t + b_y ]

* (x_t) = input at time (t)
* (h_t) = hidden state (memory)
* (y_t) = output (e.g., logits over vocabulary)

**Training** uses backpropagation through time (BPTT): unroll the recurrence across time steps and apply backprop.

### Simple intuition / example

Think of (h_t) as a rolling summary. Each step updates that summary with the new input.

**Limitation:** gradients through many time steps can vanish or explode, so long-range dependencies are hard to learn.

---

## 4) LSTM (Long Short-Term Memory)

### What

An RNN variant with an internal cell state and gates that control information flow.

### Why

Designed to solve the vanishing gradient problem and allow the network to remember or forget information over long ranges.

### How (gates & equations)

At time (t):

[ f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) ] (forget gate)
[ i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) ] (input gate)
[ \tilde{c}*t = \tanh(W_c x_t + U_c h*{t-1} + b_c) ] (candidate cell)
[ c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}*t ] (cell state)
[ o_t = \sigma(W_o x_t + U_o h*{t-1} + b_o) ] (output gate)
[ h_t = o_t \odot \tanh(c_t) ]

* Gates are element-wise vectors between 0 and 1 (sigmoid) that decide what to keep, update, and expose.

### Intuition

* **Forget gate** (f_t) decides which parts of previous cell (c_{t-1}) to keep.
* **Input gate** (i_t) decides how much new information to write.
* **Cell state** (c_t) acts like a conveyor belt — information can flow unchanged across many steps.

This architecture helps gradients flow more stably through long sequences.

---

## 5) Sequence-to-Sequence (Seq2Seq)

### What

A general pattern: an **encoder** reads the input sequence and produces a representation; a **decoder** consumes that representation to produce an output sequence.

Typical use: machine translation, summarization, speech recognition.

### Why

Before attention, decoders had to compress the entire input into a single fixed-size vector (the final hidden state of the encoder). That bottleneck limited performance on long inputs.

### How

* **Encoder:** reads input tokens and builds a final state (or sequence of states).
* **Decoder:** starts from that state and generates tokens one by one.
* **Teacher forcing:** during training, decoder receives the ground-truth previous token as input to the next step (helps training stability).
* **Inference:** either greedy decoding (pick highest-prob token each step), sampling (stochastic), or beam search (keeps top-k sequences).

**Limitations:** fixed-size bottleneck (fixed vector) forces loss of fine-grained info for long sequences.

---

## 6) Attention

### Intuition

Instead of forcing the model to compress the entire input into a single vector, let the decoder look back at the encoder outputs and **attend** to the most relevant parts for each output token.

### Basic mechanics (conceptual)

At each decoder step, compute a set of scores between the decoder state and each encoder output. Convert scores to weights (softmax) and take a weighted sum of encoder outputs — this is the context vector used to predict the next token.

### Two common formulations

* **Additive (Bahdanau) attention:** score = a(s_{t-1}, h_j) where a is a small feed-forward network.
* **Scaled dot-product attention (used by Transformers):**

[ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V ]

where Q (queries), K (keys), and V (values) are matrices derived from hidden states.

**Why scale by (\sqrt{d_k})?** To avoid extremely large dot-products when vector dimensions are large, which would push softmax into regions with very small gradients.

### Example intuition

If the decoder wants to produce the word "apples", it might place high attention weight on encoder tokens like "fruit" or "oranges" depending on context.

Attention solves the fixed-bottleneck problem and provides alignment-style behavior.

---

## 7) Transformers (the big leap)

### Why attention-only?

* Recurrence is sequential and hard to parallelize across time steps.
* Attention compares all pairs of positions directly, enabling massive parallelism on GPUs/TPUs.
* Attention can capture long-range dependencies easily.

**Key idea:** replace recurrence with stacks of attention and position-wise feed-forward layers.

### Transformer block (one layer) — high level

1. **Multi-head Self-Attention** (each token attends to every token in the sequence)

   * Project inputs into multiple Q,K,V heads
   * For each head: compute scaled dot-product attention
   * Concatenate outputs and do a final linear projection
2. **Add & Norm** (residual connection + layer normalization)
3. **Position-wise Feed-Forward Network (FFN)**

   * A small 2-layer MLP applied independently to each position
4. **Add & Norm** again

Stack many such layers (6, 12, 24, ...). Use different stacks for encoder and decoder in encoder-decoder Transformers.

### Positional encoding

Because attention has no recurrence or convolution, we add a positional signal so the model knows token order.

Common choice (sinusoidal):

[ PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) ]
[ PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) ]

These add to token embeddings so position information is injected.

### Why multi-head?

Each head can focus on different types of relationships (syntax, co-reference, locality, etc.). Multi-head attention captures diverse signals in parallel.

### Training & scaling

Transformers allowed training on huge corpora efficiently because of parallelization. That led to the pretrain-then-finetune paradigm and, eventually, the LLM era where larger models tended to perform better.

---

## 8) GPT vs BERT (architectural & objective differences)

### Architectures

* **GPT (decoder-only):** stack of transformer *decoder* blocks with causal (auto-regressive) masking so each position can only attend to previous positions. Used for text generation (predict next token). Good at tasks needing generation.

* **BERT (encoder-only):** stack of transformer *encoder* blocks. Bidirectional (each token can attend to both left and right) — trained with masked language modeling (some tokens masked; model predicts them). Good at classification, QA (extractive), and other understanding tasks.

* **Encoder-Decoder Transformers (e.g., original Transformer for translation, T5):** encoder encodes input, decoder generates output while attending to encoder outputs.

### Training objectives

* **GPT:** causal language modeling. Maximize probability of next token given previous tokens.

* **BERT:** masked language modeling (MLM); optionally next sentence prediction (NSP) in the original paper. Recent pretrained models often use variations like replaced token detection.

### Which to use when

* **Generation (dialog, summarization, code generation):** decoder-only (GPT-style) or encoder-decoder (T5/BART).
* **Understanding (classification, NER, QA extractive):** encoder-only (BERT-style) or encoder output used as features.

---

## 9) Practical details & gotchas

### Tokenization

* Modern models use subword tokenizers: BPE, WordPiece, SentencePiece. Subword tokens balance vocabulary size and OOV handling.

### Loss & evaluation

* Language models: cross-entropy loss on token prediction and perplexity for evaluation.
* Downstream tasks: accuracy, F1, BLEU (for translation), ROUGE (for summarization), etc.

### Compute and data

* Transformers scale well with data, model size, and compute. More data + bigger models generally improve results but at increasing cost.

### Limitations

* **Quadratic attention cost:** attention computes pairwise scores between all tokens → O(n^2) memory and compute. For very long documents, use sparse or linear attention variants.
* **Hallucinations:** generative models can produce plausible but incorrect facts.
* **Bias & safety:** models inherit biases from training data.

### Common fixes / extensions

* **Efficient attention variants:** Longformer, Reformer, Performer, etc.
* **Retrieval-augmented generation (RAG):** fetch documents from a datastore and condition the model on them to reduce hallucination.
* **Fine-tuning vs LoRA / adapters:** parameter-efficient ways to adapt large models.

---

## 10) Short code snippets (pseudocode / PyTorch-like for intuition)

### Tiny RNN step (conceptual Python)

```python
# pseudo-code (not optimized)
h = zeros(hidden_size)
for x in seq:
    h = tanh(Wxh @ x + Whh @ h + bh)
    y = Why @ h + by
```

### Scaled dot-product attention (matrix form)

```python
# Q, K, V: (seq_len, d_k)
scores = Q @ K.T                 # (seq_len, seq_len)
scaled = scores / math.sqrt(d_k)
weights = softmax(scaled, dim=-1)
output = weights @ V             # (seq_len, d_v)
```

### Transformer block (very simplified)

```python
# x: (batch, seq_len, d_model)
attn_out = MultiHeadAttention(x, x, x)  # self-attention
x = LayerNorm(x + attn_out)
ffn_out = FeedForward(x)
x = LayerNorm(x + ffn_out)
```

---

## 11) Glossary (short)

* **Embedding:** vector representation for tokens.
* **Logits:** raw model outputs before softmax.
* **Softmax:** converts logits to probabilities.
* **Perplexity:** exponentiated cross-entropy; lower is better.
* **Teacher forcing:** training trick in seq2seq where true previous token is fed to decoder during training.
* **Beam search:** keeps top-k sequences during generation.

---

## 12) Where to go next (recommended learning path)

1. Implement a simple RNN / LSTM for next-character prediction.
2. Build a Seq2Seq model for tiny translation (toy dataset) with and without attention.
3. Read and implement scaled dot-product attention.
4. Read the Transformer paper and implement a minimal transformer block.
5. Try using transformer libraries (Hugging Face Transformers) to fine-tune a small pretrained model.

---

## 13) References / Seminal papers (titles only — easy to search)

* Hochreiter & Schmidhuber — *Long Short-Term Memory* (1997)
* Sutskever, Vinyals, Le — *Sequence to Sequence Learning with Neural Networks* (2014)
* Bahdanau, Cho, Bengio — *Neural Machine Translation by Jointly Learning to Align and Translate* (2014)
* Vaswani et al. — *Attention is All You Need* (2017)
* Devlin et al. — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018)
* Radford et al. — *Improving Language Understanding by Generative Pre-Training* (GPT family papers)

---

## Closing note

This README is meant to be both a conceptual map and a practical launching pad. Each section contains "what / why / how / when" so you can quickly understand the purpose of each idea and how it fixed shortcomings of previous approaches.

If you want, I can also:

* Produce a one-page cheat sheet with diagrams.
* Produce runnable Jupyter notebooks for: (a) char-level RNN, (b) Seq2Seq + attention, (c) mini-transformer.
* Create slides for a 15–20 minute talk.

Tell me which follow-up you'd like and I will generate it next.

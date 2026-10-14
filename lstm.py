import numpy as np
import re
import pickle
from collections import Counter
from typing import List, Tuple, Optional


def sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def tanh(x):
    return np.tanh(x)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


def clip_gradients(grads: List[np.ndarray], max_norm: float) -> float:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")
    total_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for g in grads:
            g *= scale
    return float(total_norm)


def relative_error(a: float, b: float) -> float:
    denom = max(1e-12, abs(a) + abs(b))
    return abs(a - b) / denom


def gradients_match(numerical: float, analytical: float,
                    atol: float = 1e-7, rtol: float = 1e-4) -> bool:
    return abs(numerical - analytical) <= (
        atol + rtol * max(abs(numerical), abs(analytical))
    )


class Embedding:
    def __init__(self, vocab_size: int, dim: int, name: str = "Embedding"):
        self.vocab_size = vocab_size
        self.dim = dim
        self.name = name
        self.W = np.random.randn(vocab_size, dim) * 0.1
        self.grad = np.zeros_like(self.W)
        self.last_indices: Optional[np.ndarray] = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self.last_indices = indices
        return self.W[indices]

    def backward(self, grad_output: np.ndarray) -> None:
        self.grad.fill(0)
        np.add.at(self.grad, self.last_indices, grad_output)

    def update(self, lr: float):
        self.W -= lr * self.grad

    def parameters(self):
        return [self.W]

    def gradients(self):
        return [self.grad]


class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, name: str = "LSTM"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.name = name

        scale = np.sqrt(1.0 / (input_dim + hidden_dim))
        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wg = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale

        self.bf = np.zeros(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)
        self.bg = np.zeros(hidden_dim)

        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWo = np.zeros_like(self.Wo)
        self.dWg = np.zeros_like(self.Wg)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbo = np.zeros_like(self.bo)
        self.dbg = np.zeros_like(self.bg)

        self.cache = {}

    def forward(self, x, h0=None, c0=None):
        batch, seq_len, _ = x.shape

        if h0 is None:
            h0 = np.zeros((batch, self.hidden_dim))
        if c0 is None:
            c0 = np.zeros((batch, self.hidden_dim))

        h = h0
        c = c0

        h_states = [h0]
        c_states = [c0]
        f_gates, i_gates, o_gates, g_gates = [], [], [], []
        z_list = []
        inputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            inputs.append(x_t)

            concat = np.concatenate([h, x_t], axis=1)
            z_list.append(concat)

            f = sigmoid(concat @ self.Wf + self.bf)
            i = sigmoid(concat @ self.Wi + self.bi)
            o = sigmoid(concat @ self.Wo + self.bo)
            g = tanh(concat @ self.Wg + self.bg)

            c = f * c + i * g
            h = o * tanh(c)

            f_gates.append(f)
            i_gates.append(i)
            o_gates.append(o)
            g_gates.append(g)
            h_states.append(h)
            c_states.append(c)

        self.cache = {
            "inputs": inputs,
            "z_list": z_list,
            "f_gates": f_gates,
            "i_gates": i_gates,
            "o_gates": o_gates,
            "g_gates": g_gates,
            "h_states": h_states,
            "c_states": c_states,
            "h0": h0,
            "c0": c0,
        }

        output = np.stack(h_states[1:], axis=1)
        return output, h, c

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = grad_output.shape

        self.dWf.fill(0)
        self.dWi.fill(0)
        self.dWo.fill(0)
        self.dWg.fill(0)
        self.dbf.fill(0)
        self.dbi.fill(0)
        self.dbo.fill(0)
        self.dbg.fill(0)

        dx = np.zeros((batch, seq_len, self.input_dim))

        dh_next = np.zeros((batch, self.hidden_dim))
        dc_next = np.zeros((batch, self.hidden_dim))

        for t in reversed(range(seq_len)):
            dh = grad_output[:, t, :] + dh_next

            c_prev = self.cache["c_states"][t]
            c_t = self.cache["c_states"][t + 1]
            z = self.cache["z_list"][t]
            f = self.cache["f_gates"][t]
            i = self.cache["i_gates"][t]
            o = self.cache["o_gates"][t]
            g = self.cache["g_gates"][t]

            tanh_c = np.tanh(c_t)

            do = dh * tanh_c
            do_pre = do * o * (1 - o)

            dc = dh * o * (1 - tanh_c ** 2) + dc_next

            df = dc * c_prev
            df_pre = df * f * (1 - f)

            di = dc * g
            di_pre = di * i * (1 - i)

            dg = dc * i
            dg_pre = dg * (1 - g ** 2)

            self.dWf += z.T @ df_pre
            self.dWi += z.T @ di_pre
            self.dWo += z.T @ do_pre
            self.dWg += z.T @ dg_pre

            self.dbf += df_pre.sum(axis=0)
            self.dbi += di_pre.sum(axis=0)
            self.dbo += do_pre.sum(axis=0)
            self.dbg += dg_pre.sum(axis=0)

            dz = (df_pre @ self.Wf.T +
                  di_pre @ self.Wi.T +
                  do_pre @ self.Wo.T +
                  dg_pre @ self.Wg.T)

            dh_next = dz[:, :self.hidden_dim]
            dx[:, t, :] = dz[:, self.hidden_dim:]

            dc_next = dc * f

        return dx

    def update(self, lr: float):
        self.Wf -= lr * self.dWf
        self.Wi -= lr * self.dWi
        self.Wo -= lr * self.dWo
        self.Wg -= lr * self.dWg
        self.bf -= lr * self.dbf
        self.bi -= lr * self.dbi
        self.bo -= lr * self.dbo
        self.bg -= lr * self.dbg

    def parameters(self):
        return [self.Wf, self.Wi, self.Wo, self.Wg,
                self.bf, self.bi, self.bo, self.bg]

    def gradients(self):
        return [self.dWf, self.dWi, self.dWo, self.dWg,
                self.dbf, self.dbi, self.dbo, self.dbg]

    def param_names(self) -> List[str]:
        return ["Wf", "Wi", "Wo", "Wg", "bf", "bi", "bo", "bg"]

    def loss(self, x, h0=None, c0=None) -> float:
        _, h_last, _ = self.forward(x, h0, c0)
        return 0.5 * float(np.sum(h_last ** 2))

    def loss_grad_output(self, x, h0=None, c0=None) -> np.ndarray:
        out, h_last, _ = self.forward(x, h0, c0)
        B, T, H = out.shape
        grad = np.zeros((B, T, H))
        grad[:, -1, :] = h_last
        return grad


class Dense:
    def __init__(self, n_in: int, n_out: int, name: str = "Dense"):
        self.name = name
        scale = np.sqrt(1.0 / n_in)
        self.W = np.random.randn(n_in, n_out) * scale
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.last_x = None

    def forward(self, x):
        self.last_x = x
        return x @ self.W + self.b

    def backward(self, grad):
        if grad.ndim == 3:
            self.dW = np.einsum('bti,btj->ij', self.last_x, grad)
            self.db = grad.sum(axis=(0, 1))
            return grad @ self.W.T
        else:
            self.dW = self.last_x.T @ grad
            self.db = grad.sum(axis=0)
            return grad @ self.W.T

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def parameters(self):
        return [self.W, self.b]

    def gradients(self):
        return [self.dW, self.db]


class PersianTokenizer:
    PUNCT_PATTERN = r'[.,!?;:،؛؟«»()\[\]{}"\'—–…\-]'
    CLOSING = set(".,!?;:،؛؟»)]}")
    OPENING = {"«", "(", "[", "{"}

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.total_unique_words = 0

        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"

    def _normalize(self, text: str) -> str:
        text = text.replace("ي", "ی").replace("ك", "ک")
        text = text.replace("ۀ", "ه").replace("أ", "ا") \
                   .replace("إ", "ا").replace("ؤ", "و")

        text = re.sub(r'[\u064B-\u0652]', '', text)
        text = text.replace("\u200d", "\u200c")
        text = text.replace("\ufeff", "")

        text = re.sub(r'(' + self.PUNCT_PATTERN + r')', r' \1 ', text)
        text = re.sub(
            r'[^\w\s\u200c' + self.PUNCT_PATTERN[1:-1] + r']',
            ' ',
            text
        )
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1.")
        counter = Counter()
        for text in texts:
            text = self._normalize(text)
            for word in text.split():
                counter[word] += 1

        self.word2idx = {self.PAD: 0, self.UNK: 1, self.BOS: 2, self.EOS: 3}

        idx = 4
        for word, freq in counter.most_common():
            if freq >= min_freq:
                self.word2idx[word] = idx
                idx += 1

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.total_unique_words = len(counter)

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        text = self._normalize(text)
        words = text.split()
        ids = [self.word2idx.get(w, self.word2idx[self.UNK]) for w in words]
        if add_special:
            ids = [self.word2idx[self.BOS]] + ids + [self.word2idx[self.EOS]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        words = []
        for i in ids:
            w = self.idx2word.get(i, self.UNK)
            if skip_special and w in (self.PAD, self.BOS, self.EOS):
                continue
            words.append(w)

        out = ""
        for i, w in enumerate(words):
            if w in self.CLOSING:
                out = out.rstrip() + w
            elif i > 0 and words[i - 1] in self.OPENING:
                out += w
            else:
                out += (" " if out else "") + w
        return out.strip()

    def count_unk(self, texts: List[str]) -> Tuple[int, int]:
        unk_id = self.word2idx[self.UNK]
        total = 0
        n_unk = 0
        for text in texts:
            ids = self.encode(text, add_special=False)
            total += len(ids)
            n_unk += sum(1 for i in ids if i == unk_id)
        return n_unk, total


class PersianLanguageModel:
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, context_len: int = 8,
                 name: str = "PersianLM"):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.name = name

        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm1 = LSTM(embed_dim, hidden_dim)
        self.lstm2 = LSTM(hidden_dim, hidden_dim)
        self.dense = Dense(hidden_dim, vocab_size)

        self.layers = [self.embedding, self.lstm1, self.lstm2, self.dense]

    def forward(self, indices: np.ndarray) -> np.ndarray:
        x = self.embedding.forward(indices)
        x, _, _ = self.lstm1.forward(x)
        x, _, _ = self.lstm2.forward(x)
        logits = self.dense.forward(x)
        return logits

    def backward(self, grad_logits: np.ndarray):
        g = self.dense.backward(grad_logits)
        g = self.lstm2.backward(g)
        g = self.lstm1.backward(g)
        self.embedding.backward(g)

    def update(self, lr: float):
        for layer in self.layers:
            layer.update(lr)

    def parameters_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params.extend(layer.parameters())
            grads.extend(layer.gradients())
        return params, grads

    def snapshot_weights(self) -> List[np.ndarray]:
        return [p.copy() for p in self.parameters_grads()[0]]

    @staticmethod
    def _ce_loss_and_grad_logits(logits: np.ndarray, targets: np.ndarray,
                                 pad_idx: int):
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)

        mask = (targets_flat != pad_idx).astype(np.float64)
        mask_sum = mask.sum()
        if mask_sum == 0:
            mask_sum = 1.0

        log_probs = log_softmax(logits_flat, axis=-1)
        correct_log_probs = log_probs[np.arange(B * T), targets_flat]
        loss = -np.sum(correct_log_probs * mask) / mask_sum

        probs = np.exp(log_probs)
        grad_logits = probs.copy()
        grad_logits[np.arange(B * T), targets_flat] -= 1.0
        grad_logits *= mask[:, None]
        grad_logits /= mask_sum
        grad_logits = grad_logits.reshape(B, T, V)
        return float(loss), grad_logits

    def compute_loss_and_grad(self, indices: np.ndarray, targets: np.ndarray,
                              pad_idx: int = 0):
        logits = self.forward(indices)
        return self._ce_loss_and_grad_logits(logits, targets, pad_idx)

    def compute_loss_batched(self, indices: np.ndarray, targets: np.ndarray,
                             pad_idx: int = 0, batch_size: int = 32) -> float:
        n = len(indices)
        if n == 0:
            return 0.0
        total_loss = 0.0
        total_tokens = 0.0
        for i in range(0, n, batch_size):
            X_b = indices[i:i + batch_size]
            Y_b = targets[i:i + batch_size]
            logits = self.forward(X_b)
            B, T, V = logits.shape
            logits_flat = logits.reshape(B * T, V)
            targets_flat = Y_b.reshape(B * T)
            mask = (targets_flat != pad_idx).astype(np.float64)
            log_probs = log_softmax(logits_flat, axis=-1)
            correct_log_probs = log_probs[np.arange(B * T), targets_flat]
            total_loss += float(-np.sum(correct_log_probs * mask))
            total_tokens += float(mask.sum())
        return total_loss / max(1.0, total_tokens)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def create_training_data(tokenizer: PersianTokenizer, texts: List[str],
                         max_len: int = 10, stride: Optional[int] = None):
    if max_len < 1:
        raise ValueError("max_len must be positive.")
    if stride is None:
        stride = max(1, max_len // 2)
    if not 1 <= stride <= max_len:
        raise ValueError("stride must be between 1 and max_len.")

    X_list, Y_list = [], []
    pad_idx = tokenizer.word2idx[tokenizer.PAD]

    for text in texts:
        ids = tokenizer.encode(text, add_special=True)
        if len(ids) < 2:
            continue

        last_start = max(0, len(ids) - 1 - max_len)
        starts = list(range(0, last_start + 1, stride))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

        for start in starts:
            x = ids[start:start + max_len]
            y = ids[start + 1:start + max_len + 1]
            if len(x) < max_len:
                x = x + [pad_idx] * (max_len - len(x))
            if len(y) < max_len:
                y = y + [pad_idx] * (max_len - len(y))
            X_list.append(x)
            Y_list.append(y)

    if not X_list:
        return (np.empty((0, max_len), dtype=np.int32),
                np.empty((0, max_len), dtype=np.int32))

    return np.array(X_list, dtype=np.int32), np.array(Y_list, dtype=np.int32)


def data_summary(tokenizer: PersianTokenizer,
                 train_texts: List[str],
                 val_texts: Optional[List[str]] = None) -> None:
    print("-" * 60)
    print("Corpus Status Report")
    print("-" * 60)
    print(f"  Vocabulary size (with special tokens): {tokenizer.vocab_size}")
    print(f"  Unique words before min_freq filter: "
          f"{tokenizer.total_unique_words}")

    n_unk, total = tokenizer.count_unk(train_texts)
    train_unk_rate = n_unk / total if total > 0 else 0.0
    print(f"  <unk> in training: {n_unk}/{total} "
          f"({100.0 * train_unk_rate:.2f}%)")

    if val_texts:
        n_unk_v, total_v = tokenizer.count_unk(val_texts)
        val_unk_rate = n_unk_v / total_v if total_v > 0 else 0.0
        print(f"  <unk> in validation: {n_unk_v}/{total_v} "
              f"({100.0 * val_unk_rate:.2f}%)")
        if val_unk_rate > 0.10:
            print("  [Warning] <unk> rate in validation is above 10%. "
                  "Consider a larger corpus or lower min_freq.")
    if train_unk_rate > 0.10:
        print("  [Warning] <unk> rate in training is above 10%; "
              "lower min_freq or enlarge the corpus.")
    print("-" * 60)


def train_language_model(texts: List[str],
                         val_texts: Optional[List[str]] = None,
                         epochs: int = 200,
                         embed_dim: int = 64, hidden_dim: int = 128,
                         lr: float = 0.005, seq_len: int = 8,
                         batch_size: int = 16,
                         max_grad_norm: float = 5.0,
                         eval_batch_size: int = 64,
                         eval_every: int = 10,
                         min_freq: int = 2,
                         verbose: bool = True):
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")
    if eval_every < 1:
        raise ValueError("eval_every must be positive.")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    if min_freq < 1:
        raise ValueError("min_freq must be at least 1.")

    tokenizer = PersianTokenizer()
    tokenizer.build_vocab(texts, min_freq=min_freq)

    if verbose:
        data_summary(tokenizer, texts, val_texts)

    if tokenizer.vocab_size <= 5:
        raise ValueError(
            f"Vocabulary after applying min_freq={min_freq} has only "
            f"{tokenizer.vocab_size} tokens. Decrease min_freq "
            f"or enlarge the corpus."
        )

    X, Y = create_training_data(tokenizer, texts, max_len=seq_len)
    if len(X) == 0:
        raise ValueError("Insufficient training data.")

    X_val, Y_val = (None, None)
    if val_texts:
        X_val, Y_val = create_training_data(tokenizer, val_texts, max_len=seq_len)
        if len(X_val) == 0:
            if verbose:
                print("[Warning] Validation set is empty; skipping evaluation.")
            X_val, Y_val = None, None

    model = PersianLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        context_len=seq_len,
    )

    initial_weights = model.snapshot_weights()

    optimizer = Adam(lr=lr)
    losses = []
    val_losses = []
    pad_idx = tokenizer.word2idx[tokenizer.PAD]

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0

        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            idx = indices[i:i + batch_size]
            X_batch = X[idx]
            Y_batch = Y[idx]

            loss, grad_logits = model.compute_loss_and_grad(
                X_batch, Y_batch, pad_idx=pad_idx
            )
            model.backward(grad_logits)

            params, grads = model.parameters_grads()
            clip_gradients(grads, max_grad_norm)
            optimizer.step(params, grads)

            valid_tokens = int(np.count_nonzero(Y_batch != pad_idx))
            epoch_loss += loss * valid_tokens
            epoch_tokens += valid_tokens

        avg_loss = epoch_loss / max(1, epoch_tokens)
        losses.append(avg_loss)

        if X_val is not None and (epoch % eval_every == 0 or epoch == epochs):
            val_loss = model.compute_loss_batched(
                X_val, Y_val, pad_idx=pad_idx, batch_size=eval_batch_size
            )
            val_losses.append(val_loss)
            if verbose:
                print(f"Epoch {epoch:4d} | train_loss={avg_loss:.4f} "
                      f"| val_loss={val_loss:.4f}")
        else:
            if verbose and (epoch % eval_every == 0 or epoch == 1):
                print(f"Epoch {epoch:4d} | train_loss={avg_loss:.4f}")

    final_weights = model.snapshot_weights()
    total_delta = sum(
        float(np.linalg.norm(after - before))
        for before, after in zip(initial_weights, final_weights)
    )
    if verbose:
        print("-" * 60)
        print("Weight Change Check During Training")
        print(f"  Sum of weight change norms: {total_delta:.6e}")
        print("-" * 60)

    if not np.isfinite(total_delta):
        raise RuntimeError(
            "Weight changes contain infinite or invalid values. "
            "Likely gradient explosion or learning rate too high."
        )
    if total_delta < 1e-12:
        raise RuntimeError(
            "Model weights did not change during training. "
            "Check gradients."
        )

    return model, tokenizer, losses, val_losses


def generate_text(model: PersianLanguageModel, tokenizer: PersianTokenizer,
                  start_text: str = "", max_len: int = 15,
                  temperature: float = 0.8, top_k: int = 5) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    if max_len < 0:
        raise ValueError("max_len cannot be negative.")

    model_ids = tokenizer.encode(start_text, add_special=False)
    model_ids = [tokenizer.word2idx[tokenizer.BOS]] + model_ids

    generated = model_ids.copy()
    context_len = model.context_len

    forbidden_ids = [
        tokenizer.word2idx[tokenizer.PAD],
        tokenizer.word2idx[tokenizer.BOS],
        tokenizer.word2idx[tokenizer.UNK],
    ]

    for _ in range(max_len):
        input_ids = np.array([generated[-context_len:]])

        logits = model.forward(input_ids)
        next_logits = logits[0, -1, :].copy()

        next_logits[forbidden_ids] = -np.inf
        next_logits = next_logits / temperature

        if top_k > 0:
            finite_idx = np.where(np.isfinite(next_logits))[0]
            k = min(top_k, len(finite_idx))
            if k < len(next_logits):
                top_k_idx = finite_idx[np.argsort(next_logits[finite_idx])[-k:]]
                mask = np.full_like(next_logits, -np.inf)
                mask[top_k_idx] = next_logits[top_k_idx]
                next_logits = mask

        probs = softmax(next_logits)
        next_id = int(np.random.choice(len(probs), p=probs))

        if next_id == tokenizer.word2idx[tokenizer.EOS]:
            break

        generated.append(next_id)

    return tokenizer.decode(generated[1:])


def save_model(model: PersianLanguageModel, tokenizer: PersianTokenizer, path: str):
    state = {
        "vocab_size": model.vocab_size,
        "embed_dim": model.embed_dim,
        "hidden_dim": model.hidden_dim,
        "context_len": model.context_len,
        "embedding": model.embedding.W.copy(),
        "lstm1": {k: getattr(model.lstm1, k).copy()
                  for k in model.lstm1.param_names()},
        "lstm2": {k: getattr(model.lstm2, k).copy()
                  for k in model.lstm2.param_names()},
        "dense_W": model.dense.W.copy(),
        "dense_b": model.dense.b.copy(),
        "tokenizer_word2idx": tokenizer.word2idx,
        "tokenizer_idx2word": tokenizer.idx2word,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_model(path: str) -> Tuple[PersianLanguageModel, PersianTokenizer]:
    with open(path, "rb") as f:
        state = pickle.load(f)

    model = PersianLanguageModel(
        vocab_size=state["vocab_size"],
        embed_dim=state["embed_dim"],
        hidden_dim=state["hidden_dim"],
        context_len=state.get("context_len", 8),
    )

    model.embedding.W = state["embedding"]
    for k in model.lstm1.param_names():
        setattr(model.lstm1, k, state["lstm1"][k])
        setattr(model.lstm2, k, state["lstm2"][k])
    model.dense.W = state["dense_W"]
    model.dense.b = state["dense_b"]

    tokenizer = PersianTokenizer()
    tokenizer.word2idx = state["tokenizer_word2idx"]
    tokenizer.idx2word = state["tokenizer_idx2word"]
    tokenizer.vocab_size = len(tokenizer.word2idx)

    return model, tokenizer


def load_corpus(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def gradient_check_lstm(seed: int = 0,
                        input_dim: int = 2,
                        hidden_dim: int = 3,
                        batch: int = 2,
                        seq_len: int = 2,
                        eps: float = 1e-5,
                        atol: float = 1e-7,
                        rtol: float = 1e-4,
                        n_per_param: Optional[int] = 5):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    lstm = LSTM(input_dim, hidden_dim)
    for name in lstm.param_names():
        p = getattr(lstm, name)
        setattr(lstm, name, rng.randn(*p.shape) * 0.5)

    x = rng.randn(batch, seq_len, input_dim)
    h0 = rng.randn(batch, hidden_dim) * 0.1
    c0 = rng.randn(batch, hidden_dim) * 0.1

    loss_val = lstm.loss(x, h0, c0)
    grad_out = lstm.loss_grad_output(x, h0, c0)
    lstm.backward(grad_out)

    print(f"[LSTM Gradient Check] input={input_dim}, hidden={hidden_dim}, "
          f"batch={batch}, seq_len={seq_len} | loss={loss_val:.6f}")
    print(f"  criterion: |num-ana| <= atol + rtol*max(|num|,|ana|) "
          f"with atol={atol:.1e}, rtol={rtol:.1e}")

    max_rel_err = 0.0
    max_abs_err = 0.0
    n_checks = 0
    n_pass = 0

    for name in lstm.param_names():
        P = getattr(lstm, name)
        dP = getattr(lstm, "d" + name)
        flat_P = P.reshape(-1)
        flat_dP = dP.reshape(-1)

        if n_per_param is None:
            idxs = np.arange(flat_P.size)
        else:
            k = min(n_per_param, flat_P.size)
            idxs = rng.choice(flat_P.size, size=k, replace=False)

        max_local_rel = 0.0
        max_local_abs = 0.0
        local_pass = 0

        for idx in idxs:
            original = flat_P[idx]
            flat_P[idx] = original + eps
            l_plus = lstm.loss(x, h0, c0)
            flat_P[idx] = original - eps
            l_minus = lstm.loss(x, h0, c0)
            flat_P[idx] = original

            num_grad = (l_plus - l_minus) / (2 * eps)
            ana_grad = flat_dP[idx]

            abs_err = abs(num_grad - ana_grad)
            rel_err = relative_error(num_grad, ana_grad)

            max_local_abs = max(max_local_abs, abs_err)
            max_local_rel = max(max_local_rel, rel_err)
            max_abs_err = max(max_abs_err, abs_err)
            max_rel_err = max(max_rel_err, rel_err)
            n_checks += 1

            if gradients_match(num_grad, ana_grad, atol=atol, rtol=rtol):
                n_pass += 1
                local_pass += 1

        print(f"  {name:3s} | max|ana|={np.max(np.abs(dP)):.3e} "
              f"| max abs err={max_local_abs:.3e} "
              f"| max rel err={max_local_rel:.3e} "
              f"| pass={local_pass}/{len(idxs)}")

    print(f"  --> {n_pass}/{n_checks} checks passed")
    print(f"  max abs err = {max_abs_err:.3e} | max rel err = {max_rel_err:.3e}")
    return max_abs_err, max_rel_err, n_pass, n_checks

def _model_named_params(model: PersianLanguageModel):
    yield "embedding.W", model.embedding.W
    for name in model.lstm1.param_names():
        yield f"lstm1.{name}", getattr(model.lstm1, name)
    for name in model.lstm2.param_names():
        yield f"lstm2.{name}", getattr(model.lstm2, name)
    yield "dense.W", model.dense.W
    yield "dense.b", model.dense.b


def _model_named_grads(model: PersianLanguageModel):
    yield "embedding.W", model.embedding.grad
    for name in model.lstm1.param_names():
        yield f"lstm1.{name}", getattr(model.lstm1, "d" + name)
    for name in model.lstm2.param_names():
        yield f"lstm2.{name}", getattr(model.lstm2, "d" + name)
    yield "dense.W", model.dense.dW
    yield "dense.b", model.dense.db


def gradient_check_model(seed: int = 0,
                         vocab_size: int = 6,
                         embed_dim: int = 3,
                         hidden_dim: int = 3,
                         batch: int = 2,
                         seq_len: int = 3,
                         eps: float = 1e-5,
                         atol: float = 1e-7,
                         rtol: float = 1e-4,
                         n_per_param: Optional[int] = 5):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    model = PersianLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        context_len=seq_len,
    )

    for _, P in _model_named_params(model):
        P *= 0.5

    X = rng.randint(0, vocab_size, size=(batch, seq_len))
    Y = rng.randint(0, vocab_size, size=(batch, seq_len))
    pad_idx = -1

    loss_val, grad_logits = model.compute_loss_and_grad(X, Y, pad_idx=pad_idx)
    model.backward(grad_logits)

    print(f"[Full Model Gradient Check] vocab={vocab_size}, embed={embed_dim}, "
          f"hidden={hidden_dim}, batch={batch}, seq_len={seq_len}")
    print(f"  loss = {loss_val:.6f}")
    print(f"  criterion: |num-ana| <= atol + rtol*max(|num|,|ana|) "
          f"with atol={atol:.1e}, rtol={rtol:.1e}")

    def L() -> float:
        return model.compute_loss_batched(
            X, Y, pad_idx=pad_idx, batch_size=batch
        )

    max_rel_err = 0.0
    max_abs_err = 0.0
    n_checks = 0
    n_pass = 0

    for (pname, P), (_, G) in zip(_model_named_params(model),
                                   _model_named_grads(model)):
        flat_P = P.reshape(-1)
        flat_G = G.reshape(-1)

        if n_per_param is None:
            idxs = np.arange(flat_P.size)
        else:
            k = min(n_per_param, flat_P.size)
            idxs = rng.choice(flat_P.size, size=k, replace=False)

        max_local_rel = 0.0
        max_local_abs = 0.0
        local_pass = 0

        for idx in idxs:
            original = flat_P[idx]
            flat_P[idx] = original + eps
            l_plus = L()
            flat_P[idx] = original - eps
            l_minus = L()
            flat_P[idx] = original

            num_grad = (l_plus - l_minus) / (2 * eps)
            ana_grad = flat_G[idx]

            abs_err = abs(num_grad - ana_grad)
            rel_err = relative_error(num_grad, ana_grad)

            max_local_abs = max(max_local_abs, abs_err)
            max_local_rel = max(max_local_rel, rel_err)
            max_abs_err = max(max_abs_err, abs_err)
            max_rel_err = max(max_rel_err, rel_err)
            n_checks += 1

            if gradients_match(num_grad, ana_grad, atol=atol, rtol=rtol):
                n_pass += 1
                local_pass += 1

        print(f"  {pname:14s} | max|ana|={np.max(np.abs(G)):.3e} "
              f"| max abs err={max_local_abs:.3e} "
              f"| max rel err={max_local_rel:.3e} "
              f"| pass={local_pass}/{len(idxs)}")

    print(f"  --> {n_pass}/{n_checks} checks passed")
    print(f"  max abs err = {max_abs_err:.3e} | max rel err = {max_rel_err:.3e}")
    return max_abs_err, max_rel_err, n_pass, n_checks

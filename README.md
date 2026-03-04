# go-microgpt

A Go port of Andrej Karpathy's [microGPT](https://karpathy.github.io/2026/02/12/microgpt/) — a minimal, self-contained character-level GPT implementation with no external ML dependencies.

The original project distills everything needed to understand a modern language model into a single file: dataset loading, tokenization, automatic differentiation, transformer architecture, training, and inference. This port faithfully reproduces that in idiomatic Go, keeping the same educational spirit while using Go's type system to make the data flow explicit.

## Try Online

- **GitHub Pages demo:** `https://meko-christian.github.io/go-microgpt/`
- **Run locally:**

```bash
just web-build
python3 -m http.server --directory web/dist 8080
# open http://localhost:8080
```

## What it does

Trains a tiny transformer on a list of words (one per line), then generates new words that follow the same statistical patterns. Out of the box it trains on [Karpathy's names dataset](https://github.com/karpathy/makemore/blob/master/names.txt) and generates hallucinated names:

```
num docs:   32033
vocab size: 28
num params: 4192
step 1000 / 1000 | loss 2.1234
--- inference (new, hallucinated names) ---
sample  1: alara
sample  2: mion
sample  3: davi
...
```

## How it works

Everything is built from scratch on top of a scalar autograd engine:

- **`autograd.go`** — `Value` nodes form a dynamic computation graph. `Backward()` topologically sorts the graph and applies the chain rule to accumulate gradients.
- **`model.go`** — GPT-style transformer: token + position embeddings, multi-head causal self-attention with KV cache, ReLU MLP, RMSNorm, residual connections, and a linear LM head. All weights are `*Value` nodes; `Vec` and `Matrix` are just `[]*Value` slices.
- **`data.go`** — Character-level tokenizer built from the training corpus. A special BOS token is used as the sequence delimiter.
- **`train.go`** — Next-token prediction with cross-entropy loss. Autoregressive sampling with temperature control.
- **`optimizer.go`** — Adam with linear learning-rate decay.

The default model has **4,192 parameters** (1 layer, 16-dim embeddings, 4 heads, block size 16).

## Usage

```bash
# Train on the default names dataset (auto-downloaded) and generate samples
go run ./cmd/microgpt

# Custom options
go run ./cmd/microgpt \
  --input names.txt \
  --steps 2000 \
  --samples 20 \
  --temperature 0.8 \
  --n-layer 2 \
  --n-embd 32 \
  --n-head 4 \
  --block-size 32 \
  --seed 42

# Build a binary
just build
./bin/microgpt --steps 500
```

All flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `input.txt` | Training file (one document per line) |
| `--steps` | `1000` | Training steps |
| `--temperature` | `0.5` | Sampling temperature (0, 1] |
| `--samples` | `20` | Number of generated samples |
| `--seed` | `42` | RNG seed |
| `--n-layer` | `1` | Transformer layers |
| `--n-embd` | `16` | Embedding dimension (must be divisible by `--n-head`) |
| `--n-head` | `4` | Attention heads |
| `--block-size` | `16` | Maximum context length |

## Development

Requires [just](https://github.com/casey/just) and [golangci-lint](https://golangci-lint.run/).

```bash
just test      # Run tests
just lint      # Lint
just ci        # Full check: format + test + lint + mod tidy
```

## Credits

Original concept and Python implementation by [Andrej Karpathy](https://karpathy.ai/) — see the blog post [microGPT](https://karpathy.github.io/2026/02/12/microgpt/) for a detailed walkthrough of the ideas behind the code.

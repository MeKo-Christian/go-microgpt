# AGENTS.md

This file provides guidance to AI agents (Claude Code, Codex etc.) when working with code in this repository.

## Commands

All tasks are managed via `just` (requires [just](https://github.com/casey/just) and [treefmt](https://github.com/numtide/treefmt)):

```bash
just build          # Build binary to bin/microgpt
just test           # Run all tests
just lint           # Run golangci-lint
just lint-fix       # Run golangci-lint with auto-fix
just fmt            # Format source files
just run            # Run the CLI (e.g. just run --steps 50 --samples 5)
just ci             # Full CI: check-formatted, test, lint, check-tidy
just clean          # Remove build artifacts and caches
```

Run a single test:
```bash
GOCACHE="$(pwd)/.cache/go-build" go test -run TestName ./...
```

Build and run directly:
```bash
go run ./cmd/microgpt --steps 100 --samples 10
```

## Architecture

This is a single-package Go project (`package microgpt`) implementing a character-level GPT from scratch — no ML frameworks. The CLI lives in `cmd/microgpt/main.go`.

**Computation layer (`autograd.go`)**: `Value` is a scalar node in a dynamic computation graph. Each node stores its `Data`, accumulated `Grad`, parent `children []*Value`, and `localGrads []float64` (∂output/∂child). `Backward()` builds a topological sort and propagates gradients in reverse. All neural network ops (`Add`, `Mul`, `Pow`, `Exp`, `Log`, `ReLU`, etc.) are defined here.

**Model (`model.go`)**: Transformer architecture with multi-head self-attention and MLP blocks. Key types:
- `Vec = []*Value`, `Matrix = []Vec` — all weights are `*Value` nodes
- `ModelConfig` holds `NLayer`, `NEmbd`, `BlockSize`, `NHead`
- `LayerParams` holds the 6 weight matrices per layer (QKV projections, output, two MLP FC layers)
- `KVCache` accumulates keys/values during autoregressive generation
- `ForwardToken(tokenID, posID, cache)` runs one token through the full transformer, returning logit `Vec`
- Normalization is RMSNorm (no learned scale/bias); no positional encoding beyond learned `WPE`

**Data (`data.go`)**: Character-level `Tokenizer` built from the training corpus. BOS token = `len(chars)`. `LoadDocs` auto-downloads Karpathy's `names.txt` if the input file is missing.

**Training (`train.go`)**: `Train` runs next-token prediction with cross-entropy loss and calls `Backward()` + Adam each step. `GenerateSamples` does autoregressive sampling with temperature.

**Optimizer (`optimizer.go`)**: Standard Adam with linear learning-rate decay.

## Key constraints

- `NEmbd` must be divisible by `NHead`
- `gosec` and `revive` linters are disabled (see `.golangci.yml`)
- Build/lint caches are stored in `.cache/` (local to the repo)

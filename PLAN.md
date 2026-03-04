# Performance Optimization Plan: go-microgpt

Based on techniques from [eemicrogpt](https://github.com/Entrpi/eemicrogpt) — a C implementation of
the same architecture achieving up to 19,000x speedup over CPython via explicit gradients, batched
GEMM, NEON SIMD, and fused operations.

## Summary of Gains from eemicrogpt

| Technique                          | eemicrogpt gain | Notes                              |
| ---------------------------------- | --------------- | ---------------------------------- |
| Explicit fwd/bwd (no autograd)     | ~50–200x        | Biggest win — no graph overhead    |
| float32 instead of float64         | 2x memory BW    | 2x wider SIMD                      |
| Batch size 16                      | ~16x            | Amortizes weight loads             |
| Padding skip                       | 41%             | Names avg 6.5 chars vs 16 max      |
| NEON SIMD (4-wide float32)         | 18%             | Every inner loop vectorized        |
| Register-accumulator weight grads  | 23%             | Avoids load/store in inner loop    |
| Transposed weight copies           | 6%              | Cache-friendly backward matvecs    |
| 2-position batched linear          | 9%              | Halves weight bandwidth in fwd     |
| Fast vectorized exp (degree-3)     | 11%             | 1e-4 error, fine for softmax       |
| Fused linear+relu, linear+residual | 5%              | Eliminates intermediate allocations|
| Operation-sequential forward       | 10%             | Keeps weights L1-resident at d≥64  |
| AVX2 (8-wide float32, x86-64)      | ~2x over NEON   | Not in eemicrogpt — we add this    |
| Goroutine parallelism (batch dim)  | ~Ncpu×          | Not in eemicrogpt — we add this    |

---

## Architecture Overview

The current code uses a scalar autograd `Value` graph (float64). Every operation allocates a
heap node. The backward pass traverses an O(N) DAG. This is intentionally educational but 50–200×
slower than explicit analytic gradients.

### Target Architecture

```
tensor_model.go          — float32 weight arrays, explicit forward/backward, batched training
tensor_train.go          — batched train loop, gradient accumulation, Adam step
ops.go                   — pure-Go implementations of all ops (reference + fallback)
simd_amd64.go            — AMD64 function stubs (build constraint: amd64)
simd_amd64.s             — AVX2 Plan 9 assembly (amd64)
simd_arm64.go            — ARM64 function stubs (build constraint: arm64)
simd_arm64.s             — NEON Plan 9 assembly (arm64)
simd_generic.go          — portable fallbacks (build constraint: !amd64,!arm64)
```

The existing `autograd.go` / `model.go` / `train.go` remain unchanged — they serve as the
correctness reference and continue to be used by the web WASM demo. New code lives in a
parallel `tensor_*` namespace.

---

## Phase 0: Foundational Refactor — Explicit Float32 Tensors

**Goal**: Replace the autograd graph with flat `[]float32` arrays and analytic gradients.
This alone is expected to yield 50–200× speedup.

### 0.1 — Model struct

```go
// tensor_model.go

type TensorConfig struct {
    DModel    int   // embedding dim (default 64)
    NHeads    int   // attention heads (default 4)
    DFF       int   // FFN hidden dim (= 4 * DModel)
    VocabSize int   // 27 for names dataset
    BlockSize int   // max sequence length (16)
    Batch     int   // training batch size (16)
}

func (c TensorConfig) HeadDim() int { return c.DModel / c.NHeads }
```

Flat weight arrays, row-major `W[outDim][inDim]`:

```go
type TensorModel struct {
    Cfg  TensorConfig
    // Weights (flat float32, row-major)
    TokEmb []float32  // [VocabSize][DModel]
    PosEmb []float32  // [BlockSize][DModel]
    Wq     []float32  // [DModel][DModel]
    Wk     []float32  // [DModel][DModel]
    Wv     []float32  // [DModel][DModel]
    Wo     []float32  // [DModel][DModel]
    Wf1    []float32  // [DFF][DModel]
    Wf2    []float32  // [DModel][DFF]
    Wlm    []float32  // [VocabSize][DModel]
    // Transposed copies for backward matvecs (not updated by Adam)
    WqT    []float32  // [DModel][DModel]
    WkT    []float32  // [DModel][DModel]
    WvT    []float32  // [DModel][DModel]
    WoT    []float32  // [DModel][DModel]
    Wf1T   []float32  // [DModel][DFF]
    Wf2T   []float32  // [DFF][DModel]
}
```

### 0.2 — Activation cache (forward only, one batch at a time)

```go
type TensorCache struct {
    // [Batch][BlockSize][DModel] stored flat as [Batch*BlockSize*DModel]
    RawEmb      []float32
    Emb         []float32
    EmbRMS      []float32  // [Batch][BlockSize]
    Norm1       []float32
    Norm1RMS    []float32
    Q, K, V     []float32
    AttnScores  []float32  // [Batch][NHeads][BlockSize][BlockSize]
    AttnOut     []float32
    Res1        []float32
    Norm2       []float32
    Norm2RMS    []float32
    FfPreReLU   []float32  // [Batch][BlockSize][DFF]
    FfHidden    []float32
    Res2        []float32
    Logits      []float32  // [Batch][BlockSize][VocabSize]
    Probs       []float32
    SeqLens     []int      // [Batch] — actual lengths, for padding skip
    Tokens      []int      // [Batch][BlockSize]
}
```

### 0.3 — Gradient struct (mirrors model, excludes transposed copies)

```go
type TensorGrads struct {
    TokEmb []float32
    PosEmb []float32
    Wq, Wk, Wv, Wo   []float32
    Wf1, Wf2          []float32
    Wlm               []float32
}
```

### 0.4 — Explicit forward pass

Implement `Forward(m *TensorModel, c *TensorCache) float32`:

```
For each batch item b:
  For t in 0..SeqLens[b]-1:
    RawEmb[b][t] = TokEmb[tokens[b][t]] + PosEmb[t]
    Emb[b][t]    = RMSNorm(RawEmb[b][t])         // stores EmbRMS[b][t]
    Norm1[b][t]  = RMSNorm(Emb[b][t])            // stores Norm1RMS[b][t]
    Q[b][t]      = Wq @ Norm1[b][t]
    K[b][t]      = Wk @ Norm1[b][t]
    V[b][t]      = Wv @ Norm1[b][t]
  For t in 0..SeqLens[b]-1:                       // causal attention
    scores[h][t][s] = Q[b][t][h*hd:] · K[b][s][h*hd:] / sqrt(hd)  for s<=t
    AttnScores[b][h][t] = softmax(scores[h][t][:t+1])
    AttnOut[b][t][h*hd:] = sum_s AttnScores*V[b][s][h*hd:]
  For t in 0..SeqLens[b]-1:
    Res1[b][t]   = Emb[b][t] + Wo @ AttnOut[b][t]
    Norm2[b][t]  = RMSNorm(Res1[b][t])            // stores Norm2RMS[b][t]
    FfPreReLU[b][t] = Wf1 @ Norm2[b][t]
    FfHidden[b][t]  = ReLU(FfPreReLU[b][t])
    Res2[b][t]   = Res1[b][t] + Wf2 @ FfHidden[b][t]
    Logits[b][t] = Wlm @ Res2[b][t]
    Probs[b][t]  = softmax(Logits[b][t])
Cross-entropy loss over (b,t) pairs where t < SeqLens[b]-1
```

### 0.5 — Explicit backward pass (3-pass structure from eemicrogpt)

`Backward(m *TensorModel, c *TensorCache, g *TensorGrads)`:

**Pass 1 — Input gradient matvecs (position-sequential, last→first):**
For each batch item, for each position t (reverse):
- LM head backward: `dlogits[v] = probs[v] - (1 if target==v else 0)` (scaled by 1/count)
  - `g.Wlm[v] += dlogits[v] * Res2[b][t]` (weight grad, include in pass 1 for LM head)
  - `dx = Wlm^T @ dlogits` — uses transposed weight
- FFN backward: `dFfHidden = WoT @ dx`; apply ReLU mask; `dNorm2 = Wf1T @ dFfHidden`
- RMSNorm2 backward → update dx
- O projection backward: `dAttnOut = WoT @ dx`
- Attention backward: softmax-Jacobian, scatter into dQ/dK/dV

**Pass 2 — Weight gradients (register-accumulator outer products):**
Loop over output-dim d, keep grad accumulators in registers, loop over positions:
- `g.Wq[d] += sum_t dQ[t][d] * Norm1[b][t]` (and same for Wk, Wv)
- `g.Wo[d] += sum_t dx_all[t][d] * AttnOut[b][t]`
- `g.Wf2[d] += sum_t dff_out[t][d] * FfHidden[b][t]`
- `g.Wf1[j] += sum_t dff_hidden[t][j] * Norm2[b][t]`

**Pass 3 — QKV input grads + norm/embedding backward:**
- `dNorm1 = Wq^T @ dQ + Wk^T @ dK + Wv^T @ dV`
- RMSNorm1 backward → dEmb
- Add residual: `dEmb += dx_all[t]`
- RMSNormEmb backward → dRaw
- Scatter to `g.TokEmb[token] += dRaw` and `g.PosEmb[t] += dRaw`

### 0.6 — Batched Adam

```go
func AdamStep(params, grads, m1, m2 []float32, step int, lr, b1, b2, eps float32)
```

Vectorized (see Phase 2/3 for SIMD): `param -= lr*m1/(sqrt(m2)+eps)` with bias correction.

### 0.7 — Batched train loop

```go
func TensorTrain(m *TensorModel, tokenizer *Tokenizer, docs []string, opts TrainOptions, ...) error
```

- Sample a random batch of `opts.Batch` (default 16) documents each step
- Call `Forward` → `Backward` → `AdamStep` → sync transposed weights
- Report loss (averaged over batch)

### 0.8 — Testing / validation

- Run existing autograd `Train` for 100 steps and `TensorTrain` for 100 steps with the same RNG seed
- Assert loss curves agree within numerical tolerance (float32 vs float64 difference expected)
- Benchmark both: `go test -bench=BenchmarkTrain -benchtime=5s ./...`

---

## Phase 1: Pure Go Optimizations

No assembly required — significant gains from algorithm-level changes.

### 1.1 — Core GEMV with 2-position batching

```go
// MatVec: y[j] = sum_i W[j*inDim+i] * x[i]  for j in 0..outDim-1
func MatVecF32(y, W, x []float32, outDim, inDim int)

// MatVec2: same weight, two input vectors — halves weight bandwidth
func MatVec2F32(y0, y1, W, x0, x1 []float32, outDim, inDim int)
```

Apply `MatVec2` to QKV projections (2 positions share weight loads), O projection,
FFN expand, FFN contract.

### 1.2 — Padding skip

In `Forward` and `Backward`, iterate `t` up to `SeqLens[b]` (actual length), not `BlockSize`.
The average name is ~6.5 chars → seq_len ~8 → ~50% of positions are padding.
Expected gain: ~41% (same as eemicrogpt).

### 1.3 — Transposed weight copies

After each `AdamStep`, call `syncTransposedWeights(m)`:

```go
for i in 0..DModel-1:
    for j in 0..DModel-1:
        m.WqT[j*DModel+i] = m.Wq[i*DModel+j]
        // same for Wk, Wv, Wo
for i in 0..DFF-1:
    for j in 0..DModel-1:
        m.Wf1T[j*DFF+i] = m.Wf1[i*DModel+j]
// etc.
```

This makes backward matvecs (`W^T @ dOut`) use contiguous row access instead of
stride-DModel column access → cache-friendly.

### 1.4 — Fast exp approximation

Degree-3 minimax polynomial for `exp(x)`. Relative error ~1e-4 — acceptable for
softmax (only ratios matter). Go scalar version first, later vectorized in assembly:

```go
func fastExpF32(x float32) float32 {
    x *= 1.442695041  // x * log2(e)
    x = clamp(x, -126, 126)
    xi := int32(math.Floor(float64(x)))
    r := x - float32(xi)
    // 2^integer via IEEE754 exponent bit-shift
    pow2i := math.Float32frombits(uint32(xi+127) << 23)
    // 2^fractional via polynomial: 1 + 0.693*r + 0.240*r^2 + 0.056*r^3
    p := float32(0.0558011) * r
    p = (0.2402265 + p) * r
    p = (0.6931472 + p) * r
    p = 1.0 + p
    return pow2i * p
}
```

### 1.5 — Fused operations

Implement in `ops.go`:

```go
// linear + relu in one pass (no intermediate allocation)
func MatVecReLUF32(preRelu, hidden, W, x []float32, outDim, inDim int)

// linear + add base vector (residual connection) in one pass
func MatVecResidualF32(y, base, W, x []float32, outDim, inDim int)
```

### 1.6 — Operation-sequential forward (d ≥ 64)

When `DModel >= 64`, the combined weight matrices exceed typical L1 (~64KB). Instead
of processing all operations for batch item 0 then batch item 1 (batch-sequential),
restructure as:

```
Loop: embeddings for all batch items      (no weight matrices)
Loop: QKV for all batch items             (Wq/Wk/Wv stay L1-hot)
Loop: attention for all batch items       (no weight matrices)
Loop: O projection for all batch items   (Wo stays L1-hot)
Loop: pre-FFN norm for all batch items   (no weight matrices)
Loop: FFN for all batch items             (Wf1/Wf2 stay L1-hot)
Loop: LM head for all batch items         (Wlm stays L1-hot)
```

Enable when `DModel >= 64`. Expected gain: ~10% (same as eemicrogpt at d64).

### 1.7 — float32 Adam in pure Go

Replace existing `math.Sqrt(float64(...))` with `float32` arithmetic and
`math.Float32frombits` / `bits.UintSize` tricks. Eliminates float64↔float32
conversions on every parameter update.

---

## Phase 2: Plan 9 Assembly — AMD64 (AVX2)

**Files**: `simd_amd64.go` (stubs + build tag), `simd_amd64.s` (implementation)

AVX2 processes 8 `float32` per instruction using 256-bit YMM registers.
Key instruction: `VFMADD231PS` (fused multiply-add, 8-wide).

### Assembly design rules (from go-plan9-assembly skill)

- Use `//go:build amd64` in the `.go` stub file
- Filename `simd_amd64.s` selects AMD64 automatically
- All function symbols: `·FuncName(SB)` (middle dot U+00B7)
- Args via named FP references: `a+0(FP)`, `n+16(FP)`, `ret+24(FP)`
- Frame/arg sizes in `TEXT ·Func(SB), NOSPLIT, $0-N` must be exact
- End every TEXT with `RET`
- Keep locals pointer-free (no GC issues)
- Verify offsets with `go tool objdump` after build

### 2.1 — `DotF32` (dot product, backbone of all matvecs)

```go
// simd_amd64.go
//go:noescape
func DotF32(a, b *float32, n int) float32
```

```asm
// simd_amd64.s
// func DotF32(a, b *float32, n int) float32
// Arg layout (ABI0): a=0(FP) b=8(FP) n=16(FP) ret=24(FP)
TEXT ·DotF32(SB), NOSPLIT, $0-28
    MOVQ a+0(FP), SI
    MOVQ b+8(FP), DI
    MOVQ n+16(FP), CX
    VXORPS Y0, Y0, Y0          // acc[0..7] = 0
    VXORPS Y1, Y1, Y1          // acc2 for 2-way unrolling
loop8:
    CMPQ CX, $16
    JL loop8_single
    VMOVUPS   (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS   (DI), Y2, Y0   // Y0 += Y2 * mem[DI]
    VFMADD231PS 32(DI), Y3, Y1
    ADDQ $64, SI
    ADDQ $64, DI
    SUBQ $16, CX
    JMP loop8
loop8_single:
    CMPQ CX, $8
    JL tail
    VMOVUPS (SI), Y2
    VFMADD231PS (DI), Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $8, CX
    JMP loop8_single
tail:
    TESTQ CX, CX
    JZ done
    // scalar cleanup for remaining < 8 elements
    VXORPS X2, X2, X2
tail_loop:
    MOVSS (SI), X3
    MOVSS (DI), X4
    VFMADD231SS X4, X3, X2
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ tail_loop
    VADDSS X2, X0, X0          // fold scalar tail into acc
done:
    VADDPS Y0, Y1, Y0          // combine 2 accumulators
    // horizontal sum Y0: vhaddps x3, then scalar extracts
    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    MOVSS X0, ret+24(FP)
    VZEROUPPER
    RET
```

### 2.2 — `MatVecF32` (outer loop in Go, inner dot in asm)

```go
// simd_amd64.go
// MatVecF32 computes y[j] = sum_i W[j*inDim+i] * x[i] for j in [0, outDim).
func MatVecF32(y, W, x []float32, outDim, inDim int) {
    for j := range outDim {
        y[j] = DotF32(&W[j*inDim], &x[0], inDim)
    }
}
```

Alternative: a fully-assembled matvec with the outer loop in asm too (better for
register reuse and ILP). Implement initially in Go calling `DotF32`, benchmark,
then move to full asm if bottleneck.

### 2.3 — `MatVec2F32` (2-position batched, halves weight loads)

```go
// Computes y0 = W@x0 and y1 = W@x1 in one pass over W.
//go:noescape
func MatVec2F32(y0, y1, W, x0, x1 *float32, outDim, inDim int)
```

Assembly: for each output row j, load `W[j*inDim:j*inDim+inDim]` once, FMA with both x0 and x1.
Each weight load feeds two FMAs → halves memory traffic for forward QKV/O/FFN.

### 2.4 — `RMSNormF32` (vectorized scale-by-invRMS)

```go
//go:noescape
func RMSNormScaleF32(out, x *float32, invRMS float32, n int)
```

Uses `VBROADCASTSS` to broadcast `invRMS` into YMM, then `VMULPS` in a loop.
Separate function computes sum-of-squares (for RMS calculation) using FMA.

### 2.5 — `SoftmaxF32` with fast exp

```go
//go:noescape
func FastExpF32(dst, src *float32, n int)
```

Vectorized degree-3 polynomial using AVX2 FMA. 8 exps per cycle vs. 1 with scalar `expf`.
Steps (per 8 floats):
1. `VMULPS` by `log2(e)` broadcast
2. `VROUNDPS $1` for floor (mode 1 = floor)
3. `VSUBPS` for fractional part
4. Integer exponent via `VPCVTDQ2PS` + bias 127 + shift 23 → `VCASTSI256_PS`
5. Polynomial via 3× `VFMADD231PS`
6. `VMULPS` integer × fractional parts

### 2.6 — `AdamStepF32` (fully vectorized Adam update)

```go
//go:noescape
func AdamStepF32(param, grad, m1, m2 *float32, n int, lrOverB1c, invB2c, eps float32)
```

Pre-compute `lrOverB1c = lr / (1 - beta1^step)` and `invB2c = 1 / (1 - beta2^step)` in Go.
Assembly processes 8 floats per iteration:
- `m1 = beta1*m1 + (1-beta1)*g`  → 2× FMA
- `m2 = beta2*m2 + (1-beta2)*g*g` → 2× FMA + VMULPS
- `m2hat = m2 * invB2c`
- `denom = sqrt(m2hat) + eps` → `VSQRTPS` + VADDPS
- `param -= lrOverB1c * m1 / denom` → VDIVPS + FMA

`VSQRTPS` is ~14-cycle latency on Zen4/Skylake but throughput is good when pipelined.

---

## Phase 3: Plan 9 Assembly — ARM64 (NEON)

**Files**: `simd_arm64.go`, `simd_arm64.s`

NEON processes 4 `float32` per instruction using 128-bit V registers.
Key instruction: `FMLA` (fused multiply-add, 4-wide: `Vd.4S, Vn.4S, Vm.4S`).
Go Plan 9 ARM64 syntax uses mnemonic names like `FMULAD`, `FADDV` etc.

> **Note**: SME2 (Apple M4/M5 outer-product engine) is NOT accessible from Go Plan 9
> assembly — it requires `__arm_locally_streaming` C attribute and streaming mode
> transitions that have no Plan 9 equivalent. We implement NEON (baseline ARM64 SIMD),
> which is available on all ARM64 hardware including Apple Silicon, Raspberry Pi 4+,
> and ARM servers.

### 3.1 — `DotF32` (ARM64 NEON)

```go
// simd_arm64.go
//go:noescape
func DotF32(a, b *float32, n int) float32
```

```asm
// simd_arm64.s
// func DotF32(a, b *float32, n int) float32
// ABI0: a=0(FP) b=8(FP) n=16(FP) ret=24(FP)
TEXT ·DotF32(SB), NOSPLIT, $0-28
    MOVD a+0(FP), R0
    MOVD b+8(FP), R1
    MOVD n+16(FP), R2
    VEOR V0.B16, V0.B16, V0.B16   // acc = 0
    VEOR V1.B16, V1.B16, V1.B16   // acc2 (2-way unroll)
loop4x2:
    CMP  $8, R2
    BLT  loop4
    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R1), [V4.S4]
    VFMLA V0.S4, V2.S4, V4.S4     // V0 += V2 * V4
    VLD1.P 16(R0), [V3.S4]
    VLD1.P 16(R1), [V5.S4]
    VFMLA V1.S4, V3.S4, V5.S4
    SUB  $8, R2
    B    loop4x2
loop4:
    CMP  $4, R2
    BLT  tail
    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R1), [V3.S4]
    VFMLA V0.S4, V2.S4, V3.S4
    SUB  $4, R2
    B    loop4
tail:
    CBZ  R2, done
    // scalar tail
    FMOVS $0.0, F6
tail_loop:
    FMOVS (R0), F2
    FMOVS (R1), F3
    FMADD F6, F2, F3, F6   // F6 += F2 * F3
    ADD  $4, R0
    ADD  $4, R1
    SUB  $1, R2
    CBNZ R2, tail_loop
    FMOVS F6, V0.S[0]      // fold scalar into acc
done:
    VADDV V0.S4, V0        // horizontal sum: F0 = sum(V0[0..3])
    VADD  V0.S4, V1.S4, V0.S4
    VADDV V0.S4, V0
    FMOVS F0, ret+24(FP)
    RET
```

### 3.2 — `MatVec2F32` (ARM64)

Same pattern as AMD64 but with FMLA 4-wide. One weight vector (V-register) fed into
two FMA chains simultaneously. At d=64 with 4-wide FMLA, inner loop processes
4 weight elements per cycle for 2 output positions simultaneously.

### 3.3 — `RMSNormScaleF32` (ARM64)

`VBROADCAST` equivalent on ARM64: `DUP Vd.4S, Vn.S[0]` broadcasts scalar to 4 lanes.
Then `VMUL Vd.4S, Vn.4S, Vm.4S` per group of 4 floats.

### 3.4 — `FastExpF32` (ARM64)

NEON equivalent of AVX2 fast exp:
- `FMUL V.4S, V.4S, log2e.4S`
- `FRINTM V.4S, V.4S` (floor, rounds toward −∞)
- `FSUB fractional.4S, x.4S, floor.4S`
- Integer exponent via `FCVTZS` + bias + `SHL $23` + `REINTERPRET`
- Polynomial via `FMLA` ×3

### 3.5 — `AdamStepF32` (ARM64)

NEON equivalent of AVX2 Adam:
- `FMLS` / `FMLA` for m1/m2 updates
- `FSQRT V.4S, V.4S` for sqrt
- `FADD`, `FDIV` for final parameter update

---

## Phase 4: Goroutine Parallelism (Batch Dimension)

**Goal**: Distribute the batch across CPU cores for near-linear scaling.

### 4.1 — Design

The batch is the natural parallelism axis: gradient accumulation for each batch item
is independent. Strategy:

```
WorkerPool with GOMAXPROCS goroutines
Each step:
  1. Scatter batch items to workers: worker_i gets items [i*K..(i+1)*K)
  2. Each worker: Forward + Backward into per-worker TensorCache + TensorGrads
  3. Synchronize (WaitGroup)
  4. Accumulate gradients: g_total = sum(g_worker[i]) / GOMAXPROCS  (one goroutine)
  5. AdamStep on total grads (one goroutine)
  6. syncTransposedWeights (one goroutine)
```

Per-worker gradient buffers eliminate mutex contention during backward pass.
Gradient accumulation (step 4) is a simple vector add over `len(params)` floats —
use SIMD add from Phase 2/3 here too.

### 4.2 — Memory layout

Each worker gets its own `TensorCache`. The `TensorModel` (weights) is read-only
during forward/backward and shared across workers without synchronization.
Only `TensorGrads` is per-worker (write). `TensorModel.TokEmb` / `PosEmb` are
read during embedding lookup — safe for concurrent reads.

### 4.3 — Optimal batch size with parallelism

With `P` goroutines and batch `B`, each goroutine handles `B/P` items. For P=8 cores
and B=16, each goroutine handles 2 items — low overhead. Consider batch=32 or 64
when using parallelism to maintain throughput. Add `opts.Batch` parameter (default 16,
recommended 32 for multi-core).

### 4.4 — Caveat: cache effects

With P goroutines, each touching the full weight matrix (~210KB at d=64), total L2
pressure is P× higher. At d=64 the model fits in a typical 4–32MB L2; at d=128 with
P=8 goroutines the total reads are 8×803KB ≈ 6.4MB — may spill from per-core L2 but
fits in shared L3. Benchmark to verify scaling is superlinear relative to single-core.

---

## Phase 5: Advanced Assembly Optimizations

These are high-effort items, implement only after benchmarking phases 0–4.

### 5.1 — Full matvec in assembly (outer + inner loop)

Move the outer `j` loop into assembly too. Benefits: register allocation for row
pointer arithmetic, software pipelining of load-use latency, explicit prefetch
instructions (`PREFETCHT0`).

```asm
// AMD64: prefetch next row while computing current
PREFETCHT0 64(SI)   // prefetch W[j+1] while processing W[j]
```

### 5.2 — Register-accumulator weight gradients in assembly

The Pass 2 weight gradient computation (`g.Wq[d] += sum_t dQ[t][d] * Norm1[b][t]`)
is ideal for SIMD: broadcast `dQ[t][d]` into all 8 lanes, FMA with `Norm1[b][t][k*8:k*8+8]`.
Implement the inner position loop in assembly to keep accumulator registers across iterations.

For AMD64 (d=64):
- 64/8 = 8 YMM accumulators per weight row
- Position loop: broadcast scalar + 8× VFMADD231PS
- All 16 available YMM registers used: 8 accumulators + 1 broadcast + 7 for load overlap

For ARM64 (d=64):
- 64/4 = 16 V accumulators per row — uses all 32 V registers at d=64 for QKV together
- May need to do one weight matrix at a time

### 5.3 — Transposed-layout forward for large d_model

For d=128+, the weight matrices no longer fit in L1 even one at a time. Restructure
QKV projection into column-major layout `Wq_colmaj[inDim][outDim]` so that a single
dot product reads a contiguous column. This changes the memory access pattern from
strided to contiguous for the inner loop.

Alternatively, implement a mini-GEMM with 4×4 or 8×4 output tiles that maximize
register reuse (similar to how BLIS/OpenBLAS tiles GEMM).

---

## File Layout

```
go-microgpt/
├── autograd.go              (unchanged — scalar Value autograd)
├── model.go                 (unchanged — autograd model)
├── train.go                 (unchanged — autograd train loop)
├── optimizer.go             (unchanged — autograd Adam)
├── tensor_model.go          NEW: TensorModel, TensorCache, TensorGrads structs
├── tensor_train.go          NEW: batched Forward, Backward, TensorTrain, GenerateFast
├── ops.go                   NEW: pure-Go MatVecF32, MatVec2F32, RMSNorm, FastExp, etc.
├── simd_amd64.go            NEW: AMD64 stubs (//go:build amd64)
├── simd_amd64.s             NEW: AVX2 Plan 9 assembly
├── simd_arm64.go            NEW: ARM64 stubs (//go:build arm64)
├── simd_arm64.s             NEW: NEON Plan 9 assembly
├── simd_generic.go          NEW: portable fallbacks (//go:build !amd64 && !arm64)
├── tensor_model_test.go     NEW: correctness tests vs autograd reference
├── tensor_bench_test.go     NEW: benchmarks
└── cmd/microgpt/main.go     UPDATE: add --fast flag to use TensorTrain
```

---

## Testing Strategy

### Correctness

1. **Unit tests** for each primitive: `DotF32`, `MatVecF32`, `RMSNormF32`, `FastExpF32`,
   `AdamStepF32` — compare against Go scalar reference, tolerance 1e-4 for fast exp.

2. **Forward pass parity**: run autograd `ForwardToken` vs tensor `Forward` on the same weights
   and inputs. Assert max absolute diff < 1e-5 (autograd is float64, tensor is float32).

3. **Backward pass parity**: run autograd `.Backward()` vs tensor `Backward`, compare
   gradients on each weight matrix. Expected tolerance ~1e-4 due to float32 accumulation order.

4. **Training convergence**: both paths should reach the same loss plateau within 1000 steps
   with the same RNG seed (within float32 vs float64 tolerance).

5. **Platform CI**: test AMD64 (native or via QEMU) and ARM64 separately using `GOARCH`:
   ```
   GOARCH=amd64 go test ./...
   GOARCH=arm64 go test ./...
   ```

### Performance

Run benchmarks at multiple model sizes to mirror eemicrogpt sweep:

```bash
go test -bench=BenchmarkTensorTrain -benchtime=10s -count=3 \
    -run=^$ ./... 2>&1 | tee bench.txt
benchstat bench.txt
```

Target: match or exceed eemicrogpt NEON performance per sample (d=16: <10 µs/step,
d=64: <200 µs/step at batch=16 single-threaded).

---

## Expected Speedup Progression

Starting from current go-microgpt baseline (est. ~20ms/step at d=16, batch=1):

| Phase | Change                            | Est. cumulative speedup |
| ----- | --------------------------------- | ----------------------- |
| 0     | Explicit fwd/bwd + float32        | 50–100×                 |
| 0     | Batch=16                          | additional 10–16×       |
| 1     | Padding skip + pure-Go opts       | +50%                    |
| 2/3   | SIMD (AVX2 8-wide / NEON 4-wide)  | additional 3–8×         |
| 4     | P-core goroutine parallelism      | additional P×           |
| 5     | Register-accumulator wgrads       | additional 20–30%       |

Combined target: **~1,000–10,000× faster** than current autograd baseline at d=16,
comparable to eemicrogpt C NEON but with the safety of Go's type system and GC.

---

## Implementation Order

1. `tensor_model.go` + `tensor_train.go` — Phase 0 (biggest win, pure Go)
2. `ops.go` — Phase 1 (padding skip, 2-pos batching, fast exp)
3. `simd_amd64.go` + `.s` — Phase 2 (AVX2 DotF32 first, then MatVec2)
4. `simd_arm64.go` + `.s` — Phase 3 (NEON, same interface as AMD64)
5. goroutine parallelism in `tensor_train.go` — Phase 4
6. Register-accumulator weight grads in assembly — Phase 5 (only if profiling shows bottleneck)

Always profile with `go tool pprof` and `perf stat` before moving to the next phase.

---

## Plan 9 Assembly Key Reminders (from skill)

- **File naming**: `_amd64.s` / `_arm64.s` for architecture selection
- **Symbol spelling**: `·FuncName(SB)` (U+00B7 middle dot), not `.FuncName`
- **FP references**: always named: `a+0(FP)`, never plain `0(FP)`
- **framesize-argsize**: two numbers in `$framesize-argsize`, not subtraction
- **ABI0**: stack-based, stable, used for all exported asm functions
- **NOSPLIT**: safe for leaf functions that don't grow stack
- **VZEROUPPER** on AMD64: call after any YMM use before returning to Go code
- **GC safety**: keep locals pointer-free; avoid storing Go pointers in asm locals
- **Verification**: `go test -c && go tool objdump -s '·DotF32' ./package.test`

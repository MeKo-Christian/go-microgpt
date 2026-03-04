package microgpt

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

// Forward runs the explicit forward pass for the full batch.
// Returns the average cross-entropy loss over all (batch, position) pairs
// that have a valid next-token target (i.e., positions 0..SeqLens[b]-2).
func Forward(m *TensorModel, c *TensorCache) float32 {
	cfg := m.Cfg
	B := cfg.Batch
	T := cfg.BlockSize
	D := cfg.DModel
	H := cfg.NHeads
	hd := cfg.HeadDim()
	DFF := cfg.DFF
	V := cfg.VocabSize
	invSqrtHd := float32(1.0 / math.Sqrt(float64(hd)))

	var totalLoss float32
	var totalCount int

	for b := range B {
		seqLen := c.SeqLens[b]
		if seqLen <= 0 {
			continue
		}

		// --- Embeddings, Norm1, QKV ---
		for t := range seqLen {
			tok := c.Tokens[b*T+t]
			rawOff := b*T*D + t*D
			// RawEmb = TokEmb[tok] + PosEmb[t]
			tokOff := tok * D
			posOff := t * D
			for d := range D {
				c.RawEmb[rawOff+d] = m.TokEmb[tokOff+d] + m.PosEmb[posOff+d]
			}
			// Emb = RMSNorm(RawEmb)
			c.EmbRMS[b*T+t] = RMSNormF32(
				c.Emb[rawOff:rawOff+D],
				c.RawEmb[rawOff:rawOff+D],
				D,
			)
			// Norm1 = RMSNorm(Emb)
			c.Norm1RMS[b*T+t] = RMSNormF32(
				c.Norm1[rawOff:rawOff+D],
				c.Emb[rawOff:rawOff+D],
				D,
			)
			// Q = Wq @ Norm1
			qOff := b*T*D + t*D
			MatVecF32(c.Q[qOff:qOff+D], m.Wq, c.Norm1[rawOff:rawOff+D], D, D)
			// K = Wk @ Norm1
			MatVecF32(c.K[qOff:qOff+D], m.Wk, c.Norm1[rawOff:rawOff+D], D, D)
			// V = Wv @ Norm1
			MatVecF32(c.V[qOff:qOff+D], m.Wv, c.Norm1[rawOff:rawOff+D], D, D)
		}

		// --- Causal multi-head attention ---
		for t := range seqLen {
			for h := range H {
				hs := h * hd
				qOff := b*T*D + t*D + hs

				// Compute scores for s in [0, t]
				scoreBase := b*H*T*T + h*T*T + t*T
				for s := 0; s <= t; s++ {
					kOff := b*T*D + s*D + hs
					c.AttnScores[scoreBase+s] = DotF32Go(
						c.Q[qOff:qOff+hd],
						c.K[kOff:kOff+hd],
						hd,
					) * invSqrtHd
				}
				// Softmax over [0, t+1)
				SoftmaxF32(c.AttnScores[scoreBase:scoreBase+t+1], t+1)

				// Weighted sum of V
				attnOff := b*T*D + t*D + hs
				for d := range hd {
					c.AttnOut[attnOff+d] = 0
				}
				for s := 0; s <= t; s++ {
					w := c.AttnScores[scoreBase+s]
					vOff := b*T*D + s*D + hs
					for d := range hd {
						c.AttnOut[attnOff+d] += w * c.V[vOff+d]
					}
				}
			}
		}

		// --- Output projection, FFN, LM head ---
		for t := range seqLen {
			btD := b*T*D + t*D

			// Res1 = Emb + Wo @ AttnOut (residual around attention)
			MatVecResidualF32(
				c.Res1[btD:btD+D],
				c.Emb[btD:btD+D],
				m.Wo,
				c.AttnOut[btD:btD+D],
				D, D,
			)

			// Norm2 = RMSNorm(Res1)
			c.Norm2RMS[b*T+t] = RMSNormF32(
				c.Norm2[btD:btD+D],
				c.Res1[btD:btD+D],
				D,
			)

			// FFN: FfPreReLU = Wf1 @ Norm2, FfHidden = ReLU(FfPreReLU)
			btDFF := b*T*DFF + t*DFF
			MatVecReLUF32(
				c.FfPreReLU[btDFF:btDFF+DFF],
				c.FfHidden[btDFF:btDFF+DFF],
				m.Wf1,
				c.Norm2[btD:btD+D],
				DFF, D,
			)

			// Res2 = Res1 + Wf2 @ FfHidden (residual around FFN)
			MatVecResidualF32(
				c.Res2[btD:btD+D],
				c.Res1[btD:btD+D],
				m.Wf2,
				c.FfHidden[btDFF:btDFF+DFF],
				D, DFF,
			)

			// Logits = Wlm @ Res2
			btV := b*T*V + t*V
			MatVecF32(c.Logits[btV:btV+V], m.Wlm, c.Res2[btD:btD+D], V, D)

			// Probs = softmax(Logits)
			copy(c.Probs[btV:btV+V], c.Logits[btV:btV+V])
			SoftmaxF32(c.Probs[btV:btV+V], V)

			// Cross-entropy loss for positions that predict a next token
			if t < seqLen-1 {
				target := c.Tokens[b*T+t+1]
				totalLoss += CrossEntropyF32(c.Probs[btV:btV+V], target)
				totalCount++
			}
		}
	}

	if totalCount == 0 {
		return 0
	}
	return totalLoss / float32(totalCount)
}

// ---------------------------------------------------------------------------
// Backward pass helpers
// ---------------------------------------------------------------------------

// rmsNormBackward computes the gradient through RMSNorm.
// Forward: out[i] = x[i] / rms, where rms = sqrt(mean(x^2) + eps).
// Backward: dx[i] = (dOut[i] - out[i] * dot(dOut, out) / n) / rms
// dx is written into the provided slice. dOut and out are read-only.
func rmsNormBackward(dx, dOut, out []float32, rms float32, n int) {
	dotVal := DotF32Go(dOut, out, n)
	scale := dotVal / float32(n)
	invRms := float32(1.0) / rms
	for i := range n {
		dx[i] = (dOut[i] - out[i]*scale) * invRms
	}
}

// outerProductAccum accumulates the outer product: g[j*cols+k] += a[j] * b[k]
func outerProductAccum(g, a, b []float32, rows, cols int) {
	for j := range rows {
		aj := a[j]
		off := j * cols
		for k := range cols {
			g[off+k] += aj * b[k]
		}
	}
}

// transposedMatVec computes y[k] = sum_j W[j*cols+k] * x[j] (i.e., W^T @ x)
// without an explicit transposed copy. W is [rows][cols], result is [cols].
func transposedMatVec(y, W, x []float32, rows, cols int) {
	for k := range cols {
		y[k] = 0
	}
	for j := range rows {
		xj := x[j]
		off := j * cols
		for k := range cols {
			y[k] += W[off+k] * xj
		}
	}
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

// Backward computes gradients for all model parameters.
// Assumes Forward has been called and c is populated.
func Backward(m *TensorModel, c *TensorCache, g *TensorGrads) {
	cfg := m.Cfg
	B := cfg.Batch
	T := cfg.BlockSize
	D := cfg.DModel
	H := cfg.NHeads
	hd := cfg.HeadDim()
	DFF := cfg.DFF
	V := cfg.VocabSize
	invSqrtHd := float32(1.0 / math.Sqrt(float64(hd)))

	// Count total prediction positions for loss scaling
	var totalCount int
	for b := range B {
		sl := c.SeqLens[b]
		if sl > 1 {
			totalCount += sl - 1
		}
	}
	if totalCount == 0 {
		return
	}
	invCount := float32(1.0) / float32(totalCount)

	// Temporary buffers (allocated once, reused per position)
	dLogits := make([]float32, V)
	dRes2 := make([]float32, D)
	dFfHidden := make([]float32, DFF)
	dFfPreReLU := make([]float32, DFF)
	dNorm2 := make([]float32, D)
	dRes1 := make([]float32, D)
	dAttnOut := make([]float32, D)
	dNorm1 := make([]float32, D)
	dEmb := make([]float32, D)
	dRawEmb := make([]float32, D)
	tmpD := make([]float32, D)

	// Per-batch-item Q/K/V gradient accumulators (needed for attention backward)
	dQ := make([]float32, T*D)
	dK := make([]float32, T*D)
	dV := make([]float32, T*D)

	for b := range B {
		seqLen := c.SeqLens[b]
		if seqLen <= 1 {
			continue
		}

		// Zero per-batch Q/K/V grad accumulators
		VecZeroF32(dQ, seqLen*D)
		VecZeroF32(dK, seqLen*D)
		VecZeroF32(dV, seqLen*D)

		// Accumulate dEmb per position (will be used in the second pass)
		// We need to do attention backward for all positions first since
		// dK[s] and dV[s] are scattered to from multiple t positions.
		// So we do this in two phases:
		// Phase A: LM head + FFN + O-projection backward → get dAttnOut per position,
		//          then scatter attention backward into dQ/dK/dV.
		// Phase B: QKV backward + norm1 + emb backward using accumulated dQ/dK/dV.

		// --- Phase A: reverse iterate over positions with targets ---
		// We also accumulate dRes1 contributions needed in Phase B.
		// Store dRes1 per position for Phase B.
		dRes1All := make([]float32, seqLen*D)

		for t := seqLen - 1; t >= 0; t-- {
			btD := b*T*D + t*D
			btDFF := b*T*DFF + t*DFF
			btV := b*T*V + t*V

			VecZeroF32(dRes2, D)

			// LM head backward (only for positions with targets)
			if t < seqLen-1 {
				target := c.Tokens[b*T+t+1]
				// dLogits = (Probs - one_hot(target)) * invCount
				for v := range V {
					dLogits[v] = c.Probs[btV+v] * invCount
				}
				dLogits[target] -= invCount

				// g.Wlm += dLogits outer Res2
				outerProductAccum(g.Wlm, dLogits, c.Res2[btD:btD+D], V, D)

				// dRes2 = Wlm^T @ dLogits
				// Wlm is [V][D], so Wlm^T is [D][V]. Use transposed weight if available or manual.
				transposedMatVec(dRes2, m.Wlm, dLogits, V, D)
			}

			// --- FFN backward ---
			// Res2 = Res1 + Wf2 @ FfHidden
			// dRes1 gets dRes2 via residual
			copy(dRes1, dRes2)

			// g.Wf2 += dRes2 outer FfHidden
			outerProductAccum(g.Wf2, dRes2, c.FfHidden[btDFF:btDFF+DFF], D, DFF)

			// dFfHidden = Wf2^T @ dRes2 ; Wf2 is [D][DFF], Wf2T is [DFF][D]
			MatVecF32(dFfHidden, m.Wf2T, dRes2, DFF, D)

			// Apply ReLU mask
			for j := range DFF {
				if c.FfPreReLU[btDFF+j] > 0 {
					dFfPreReLU[j] = dFfHidden[j]
				} else {
					dFfPreReLU[j] = 0
				}
			}

			// g.Wf1 += dFfPreReLU outer Norm2
			outerProductAccum(g.Wf1, dFfPreReLU, c.Norm2[btD:btD+D], DFF, D)

			// dNorm2 = Wf1^T @ dFfPreReLU ; Wf1 is [DFF][D], Wf1T is [D][DFF]
			MatVecF32(dNorm2, m.Wf1T, dFfPreReLU, D, DFF)

			// RMSNorm2 backward: dRes1 += rmsNormBwd(dNorm2, Norm2, Norm2RMS)
			rmsNormBackward(tmpD, dNorm2, c.Norm2[btD:btD+D], c.Norm2RMS[b*T+t], D)
			VecAddF32(dRes1, tmpD, D)

			// Store dRes1 for Phase B
			copy(dRes1All[t*D:(t+1)*D], dRes1)

			// --- Attention O-projection backward ---
			// Res1 = Emb + Wo @ AttnOut
			// g.Wo += dRes1 outer AttnOut
			outerProductAccum(g.Wo, dRes1, c.AttnOut[btD:btD+D], D, D)

			// dAttnOut = Wo^T @ dRes1 ; WoT is [D][D]
			MatVecF32(dAttnOut, m.WoT, dRes1, D, D)

			// --- Attention backward (per head) ---
			for h := range H {
				hs := h * hd
				scoreBase := b*H*T*T + h*T*T + t*T
				dAttnH := dAttnOut[hs : hs+hd]

				// Compute dScores[s] = dAttnH . V[b][s][h*hd:(h+1)*hd]
				// and accumulate dV[s] += scores[s] * dAttnH
				dScores := make([]float32, t+1)
				for s := 0; s <= t; s++ {
					vOff := b*T*D + s*D + hs
					dScores[s] = DotF32Go(dAttnH, c.V[vOff:vOff+hd], hd)
					// dV[s] += scores[s] * dAttnH
					w := c.AttnScores[scoreBase+s]
					for d := range hd {
						dV[s*D+hs+d] += w * dAttnH[d]
					}
				}

				// Softmax backward: dLogitsAttn[s] = scores[s] * (dScores[s] - dot(scores, dScores))
				var dotSD float32
				for s := 0; s <= t; s++ {
					dotSD += c.AttnScores[scoreBase+s] * dScores[s]
				}
				for s := 0; s <= t; s++ {
					dLogitAttn := c.AttnScores[scoreBase+s] * (dScores[s] - dotSD)
					// Scale backward: dQ[t] += dLogitAttn * K[b][s] / sqrt(hd)
					kOff := b*T*D + s*D + hs
					scaled := dLogitAttn * invSqrtHd
					for d := range hd {
						dQ[t*D+hs+d] += scaled * c.K[kOff+d]
					}
					// dK[s] += dLogitAttn * Q[b][t] / sqrt(hd)
					qOff := b*T*D + t*D + hs
					for d := range hd {
						dK[s*D+hs+d] += scaled * c.Q[qOff+d]
					}
				}
			}
		}

		// --- Phase B: QKV backward + norm1 + emb backward ---
		for t := range seqLen {
			btD := b*T*D + t*D

			// Weight gradients for Wq, Wk, Wv
			outerProductAccum(g.Wq, dQ[t*D:(t+1)*D], c.Norm1[btD:btD+D], D, D)
			outerProductAccum(g.Wk, dK[t*D:(t+1)*D], c.Norm1[btD:btD+D], D, D)
			outerProductAccum(g.Wv, dV[t*D:(t+1)*D], c.Norm1[btD:btD+D], D, D)

			// dNorm1 = WqT @ dQ + WkT @ dK + WvT @ dV
			MatVecF32(dNorm1, m.WqT, dQ[t*D:(t+1)*D], D, D)
			MatVecF32(tmpD, m.WkT, dK[t*D:(t+1)*D], D, D)
			VecAddF32(dNorm1, tmpD, D)
			MatVecF32(tmpD, m.WvT, dV[t*D:(t+1)*D], D, D)
			VecAddF32(dNorm1, tmpD, D)

			// RMSNorm1 backward: dEmb = rmsNormBwd(dNorm1, Norm1, Norm1RMS)
			rmsNormBackward(dEmb, dNorm1, c.Norm1[btD:btD+D], c.Norm1RMS[b*T+t], D)

			// Add residual from Res1 = Emb + Wo @ AttnOut → dEmb += dRes1
			VecAddF32(dEmb, dRes1All[t*D:(t+1)*D], D)

			// RMSNormEmb backward: dRawEmb = rmsNormBwd(dEmb, Emb, EmbRMS)
			rmsNormBackward(dRawEmb, dEmb, c.Emb[btD:btD+D], c.EmbRMS[b*T+t], D)

			// Scatter to TokEmb and PosEmb gradients
			tok := c.Tokens[b*T+t]
			tokOff := tok * D
			posOff := t * D
			for d := range D {
				g.TokEmb[tokOff+d] += dRawEmb[d]
				g.PosEmb[posOff+d] += dRawEmb[d]
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TensorAdam optimizer
// ---------------------------------------------------------------------------

// TensorAdam holds Adam optimizer state for flat float32 parameter arrays.
type TensorAdam struct {
	LR    float32
	Beta1 float32
	Beta2 float32
	Eps   float32
	M1    [][]float32 // first moment, per param slice
	M2    [][]float32 // second moment, per param slice
}

// NewTensorAdam allocates a new Adam optimizer for the given parameter slices.
func NewTensorAdam(paramSlices [][]float32, lr, beta1, beta2, eps float32) *TensorAdam {
	m1 := make([][]float32, len(paramSlices))
	m2 := make([][]float32, len(paramSlices))
	for i, ps := range paramSlices {
		m1[i] = make([]float32, len(ps))
		m2[i] = make([]float32, len(ps))
	}
	return &TensorAdam{
		LR:    lr,
		Beta1: beta1,
		Beta2: beta2,
		Eps:   eps,
		M1:    m1,
		M2:    m2,
	}
}

// Step performs one Adam update with bias correction and linear LR decay.
// step is 1-indexed (first call is step=1). totalSteps is used for LR decay.
func (a *TensorAdam) Step(paramSlices, gradSlices [][]float32, step, totalSteps int) {
	lrT := a.LR * (1.0 - float32(step)/float32(totalSteps))
	b1corr := float32(1.0 - math.Pow(float64(a.Beta1), float64(step)))
	b2corr := float32(1.0 - math.Pow(float64(a.Beta2), float64(step)))

	for i := range paramSlices {
		params := paramSlices[i]
		grads := gradSlices[i]
		m1 := a.M1[i]
		m2 := a.M2[i]

		for j := range params {
			g := grads[j]
			m1[j] = a.Beta1*m1[j] + (1-a.Beta1)*g
			m2[j] = a.Beta2*m2[j] + (1-a.Beta2)*g*g

			m1hat := m1[j] / b1corr
			m2hat := m2[j] / b2corr

			params[j] -= lrT * m1hat / (float32(math.Sqrt(float64(m2hat))) + a.Eps)
		}
	}
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

// TensorTrain runs the batched training loop using the explicit tensor model.
func TensorTrain(m *TensorModel, tokenizer *Tokenizer, docs []string, opts TrainOptions, progress ProgressFn) error {
	if m == nil {
		return errors.New("model must not be nil")
	}
	if tokenizer == nil {
		return errors.New("tokenizer must not be nil")
	}
	if len(docs) == 0 {
		return errors.New("docs must not be empty")
	}
	if opts.NumSteps <= 0 {
		return fmt.Errorf("num steps must be > 0, got %d", opts.NumSteps)
	}

	cfg := m.Cfg
	c := NewTensorCache(cfg)
	g := NewTensorGrads(cfg)

	paramSlices := m.ParamSlices()
	gradSlices := g.GradSlices()

	adam := NewTensorAdam(
		paramSlices,
		float32(opts.LearningRate),
		float32(opts.Beta1),
		float32(opts.Beta2),
		float32(opts.EpsAdam),
	)

	rng := rand.New(rand.NewSource(42))

	for step := range opts.NumSteps {
		// Sample batch of random documents
		for b := range cfg.Batch {
			docIdx := rng.Intn(len(docs))
			tokens := tokenizer.EncodeWithBOS(docs[docIdx])

			// Truncate to BlockSize
			seqLen := min(len(tokens), cfg.BlockSize)
			c.SeqLens[b] = seqLen

			// Copy tokens into cache, zero-pad the rest
			for t := range cfg.BlockSize {
				if t < seqLen {
					c.Tokens[b*cfg.BlockSize+t] = tokens[t]
				} else {
					c.Tokens[b*cfg.BlockSize+t] = 0
				}
			}
		}

		// Forward
		loss := Forward(m, c)

		// Zero grads
		g.Zero()

		// Backward
		Backward(m, c, g)

		// Adam step (1-indexed)
		adam.Step(paramSlices, gradSlices, step+1, opts.NumSteps)

		// Sync transposed weights for next backward pass
		syncTransposedWeights(m)

		// Report progress
		if progress != nil {
			progress(StepMetrics{
				Step:  step + 1,
				Total: opts.NumSteps,
				Loss:  float64(loss),
			})
		}
	}

	return nil
}

// ---------------------------------------------------------------------------
// Parallel training loop
// ---------------------------------------------------------------------------

// TensorTrainParallel runs batched training with goroutine parallelism.
// The batch is split across P = min(GOMAXPROCS, cfg.Batch) workers.
// Each worker has its own TensorCache and TensorGrads (no contention).
// The model weights are shared (read-only during fwd/bwd); only the main
// goroutine mutates them via Adam after gradient accumulation.
func TensorTrainParallel(m *TensorModel, tokenizer *Tokenizer, docs []string, opts TrainOptions, progress ProgressFn) error {
	if m == nil {
		return errors.New("model must not be nil")
	}
	if tokenizer == nil {
		return errors.New("tokenizer must not be nil")
	}
	if len(docs) == 0 {
		return errors.New("docs must not be empty")
	}
	if opts.NumSteps <= 0 {
		return fmt.Errorf("num steps must be > 0, got %d", opts.NumSteps)
	}

	cfg := m.Cfg

	// Determine number of workers: largest P <= GOMAXPROCS that divides Batch.
	P := runtime.GOMAXPROCS(0)
	if P > cfg.Batch {
		P = cfg.Batch
	}
	for cfg.Batch%P != 0 {
		P--
	}
	K := cfg.Batch / P // items per worker

	// Per-worker resources (allocated once, reused across steps).
	type workerState struct {
		cfg   TensorConfig
		cache *TensorCache
		grads *TensorGrads
		loss  float32
	}
	workers := make([]workerState, P)
	for i := range P {
		wcfg := cfg
		wcfg.Batch = K
		workers[i] = workerState{
			cfg:   wcfg,
			cache: NewTensorCache(wcfg),
			grads: NewTensorGrads(wcfg),
		}
	}

	// Total gradient accumulator (full-batch sized).
	totalGrads := NewTensorGrads(cfg)

	paramSlices := m.ParamSlices()
	totalGradSlices := totalGrads.GradSlices()

	adam := NewTensorAdam(
		paramSlices,
		float32(opts.LearningRate),
		float32(opts.Beta1),
		float32(opts.Beta2),
		float32(opts.EpsAdam),
	)

	rng := rand.New(rand.NewSource(42))
	invP := float32(1.0) / float32(P)

	var wg sync.WaitGroup

	for step := range opts.NumSteps {
		// Sample batch and distribute tokens across workers.
		for w := range P {
			for b := range K {
				docIdx := rng.Intn(len(docs))
				tokens := tokenizer.EncodeWithBOS(docs[docIdx])
				seqLen := min(len(tokens), cfg.BlockSize)
				workers[w].cache.SeqLens[b] = seqLen
				for t := range cfg.BlockSize {
					if t < seqLen {
						workers[w].cache.Tokens[b*cfg.BlockSize+t] = tokens[t]
					} else {
						workers[w].cache.Tokens[b*cfg.BlockSize+t] = 0
					}
				}
			}
		}

		// Parallel Forward + Backward.
		wg.Add(P)
		for w := range P {
			go func(w int) {
				defer wg.Done()
				workers[w].loss = Forward(m, workers[w].cache)
				workers[w].grads.Zero()
				Backward(m, workers[w].cache, workers[w].grads)
			}(w)
		}
		wg.Wait()

		// Accumulate loss and gradients from all workers.
		totalGrads.Zero()
		var totalLoss float32
		for w := range P {
			totalLoss += workers[w].loss
			wGradSlices := workers[w].grads.GradSlices()
			for i, gs := range wGradSlices {
				VecAddF32(totalGradSlices[i], gs, len(gs))
			}
		}
		totalLoss *= invP
		// Scale gradients by 1/P to average across workers.
		for _, gs := range totalGradSlices {
			VecScaleF32(gs, invP, len(gs))
		}

		// Adam step (1-indexed).
		adam.Step(paramSlices, totalGradSlices, step+1, opts.NumSteps)

		// Sync transposed weights for next backward pass.
		syncTransposedWeights(m)

		// Report progress.
		if progress != nil {
			progress(StepMetrics{
				Step:  step + 1,
				Total: opts.NumSteps,
				Loss:  float64(totalLoss),
			})
		}
	}

	return nil
}

// ---------------------------------------------------------------------------
// Fast generation
// ---------------------------------------------------------------------------

// GenerateFast generates text samples using the tensor model.
// Uses single-item batch (Batch=1) with the existing Forward infrastructure.
// For each new token, re-runs the full forward pass (no KV cache).
func GenerateFast(m *TensorModel, tokenizer *Tokenizer, opts SampleOptions, rng *rand.Rand) []string {
	cfg := m.Cfg
	V := cfg.VocabSize
	T := cfg.BlockSize

	// Create a temporary config with Batch=1 for generation
	genCfg := TensorConfig{
		DModel:    cfg.DModel,
		NHeads:    cfg.NHeads,
		DFF:       cfg.DFF,
		VocabSize: cfg.VocabSize,
		BlockSize: cfg.BlockSize,
		Batch:     1,
	}
	c := NewTensorCache(genCfg)

	// Build a temporary model wrapper pointing to the same weights but with Batch=1 config
	genModel := &TensorModel{
		Cfg:    genCfg,
		TokEmb: m.TokEmb,
		PosEmb: m.PosEmb,
		Wq:     m.Wq,
		Wk:     m.Wk,
		Wv:     m.Wv,
		Wo:     m.Wo,
		Wf1:    m.Wf1,
		Wf2:    m.Wf2,
		Wlm:    m.Wlm,
		WqT:    m.WqT,
		WkT:    m.WkT,
		WvT:    m.WvT,
		WoT:    m.WoT,
		Wf1T:   m.Wf1T,
		Wf2T:   m.Wf2T,
	}

	invTemp := float32(1.0 / opts.Temperature)

	samples := make([]string, 0, opts.NumSamples)
	for range opts.NumSamples {
		// Start with BOS token
		tokens := make([]int, 1, T)
		tokens[0] = tokenizer.BOS
		sample := make([]rune, 0, T)

		for pos := 0; pos < T-1; pos++ {
			seqLen := min(len(tokens), T)

			// Fill cache tokens
			c.SeqLens[0] = seqLen
			for t := range T {
				if t < seqLen {
					c.Tokens[t] = tokens[t]
				} else {
					c.Tokens[t] = 0
				}
			}

			// Run forward pass
			Forward(genModel, c)

			// Get logits from the last position
			lastPos := seqLen - 1
			logitsOff := lastPos * V

			// Apply temperature scaling and softmax
			scaledLogits := make([]float32, V)
			for v := range V {
				scaledLogits[v] = c.Logits[logitsOff+v] * invTemp
			}
			SoftmaxF32(scaledLogits, V)

			// Sample from distribution
			weights := make([]float64, V)
			for v := range V {
				weights[v] = float64(scaledLogits[v])
			}
			nextTok := SampleWeighted(rng, weights)

			// Check for end of sequence (BOS used as EOS)
			if nextTok == tokenizer.BOS {
				break
			}

			sample = append(sample, tokenizer.Chars[nextTok])
			tokens = append(tokens, nextTok)

			// Stop if we hit max length
			if len(tokens) >= T {
				break
			}
		}

		samples = append(samples, string(sample))
	}

	return samples
}

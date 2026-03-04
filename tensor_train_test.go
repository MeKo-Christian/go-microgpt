package microgpt

import (
	"math"
	"math/rand"
	"testing"
)

func smallTestConfig() TensorConfig {
	return TensorConfig{
		DModel:    16,
		NHeads:    4,
		DFF:       64,
		VocabSize: 27,
		BlockSize: 16,
		Batch:     2,
	}
}

func TestTensorModelInit(t *testing.T) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))
	m := NewTensorModel(cfg, rng)

	D := cfg.DModel
	DFF := cfg.DFF
	V := cfg.VocabSize
	B := cfg.BlockSize

	// Check weight slice sizes.
	checks := []struct {
		name string
		got  int
		want int
	}{
		{"TokEmb", len(m.TokEmb), V * D},
		{"PosEmb", len(m.PosEmb), B * D},
		{"Wq", len(m.Wq), D * D},
		{"Wk", len(m.Wk), D * D},
		{"Wv", len(m.Wv), D * D},
		{"Wo", len(m.Wo), D * D},
		{"Wf1", len(m.Wf1), DFF * D},
		{"Wf2", len(m.Wf2), D * DFF},
		{"Wlm", len(m.Wlm), V * D},
		{"WqT", len(m.WqT), D * D},
		{"WkT", len(m.WkT), D * D},
		{"WvT", len(m.WvT), D * D},
		{"WoT", len(m.WoT), D * D},
		{"Wf1T", len(m.Wf1T), D * DFF},
		{"Wf2T", len(m.Wf2T), DFF * D},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Fatalf("%s: len=%d, want %d", c.name, c.got, c.want)
		}
	}

	// Verify transposed weights are correct: WqT[j*D+i] == Wq[i*D+j].
	for i := range D {
		for j := range D {
			orig := m.Wq[i*D+j]
			trans := m.WqT[j*D+i]
			if orig != trans {
				t.Fatalf("WqT transpose mismatch at [%d][%d]: Wq=%f, WqT=%f", i, j, orig, trans)
			}
		}
	}

	// Verify Wf1T transpose: Wf1 is [DFF][D], Wf1T is [D][DFF].
	for i := range DFF {
		for j := range D {
			orig := m.Wf1[i*D+j]
			trans := m.Wf1T[j*DFF+i]
			if orig != trans {
				t.Fatalf("Wf1T transpose mismatch at [%d][%d]: Wf1=%f, Wf1T=%f", i, j, orig, trans)
			}
		}
	}
}

func TestForwardProducesValidProbs(t *testing.T) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))
	m := NewTensorModel(cfg, rng)
	c := NewTensorCache(cfg)

	// Set up a single-item batch with a short sequence.
	c.SeqLens[0] = 4
	c.SeqLens[1] = 0
	tokens := []int{26, 0, 1, 2} // BOS=26 then a, b, c
	copy(c.Tokens, tokens)

	loss := Forward(m, c)

	// Loss should be positive (cross-entropy is always > 0 for imperfect predictions).
	if loss <= 0 {
		t.Fatalf("loss should be positive, got %f", loss)
	}

	// Verify probs at the first position sum to ~1.
	V := cfg.VocabSize
	var sum float64
	for v := range V {
		p := float64(c.Probs[v])
		if p < 0 {
			t.Fatalf("prob[%d] is negative: %f", v, p)
		}
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Fatalf("probs should sum to 1, got %f", sum)
	}
}

func TestBackwardProducesGradients(t *testing.T) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))
	m := NewTensorModel(cfg, rng)
	c := NewTensorCache(cfg)
	g := NewTensorGrads(cfg)

	// Set up batch with two sequences.
	c.SeqLens[0] = 4
	c.SeqLens[1] = 3
	seqs := [][]int{
		{26, 0, 1, 2},
		{26, 3, 4},
	}
	T := cfg.BlockSize
	for b, seq := range seqs {
		for i, tok := range seq {
			c.Tokens[b*T+i] = tok
		}
	}

	Forward(m, c)
	g.Zero()
	Backward(m, c, g)

	// Verify that at least some gradients are non-zero.
	gradSlices := g.GradSlices()
	names := []string{"TokEmb", "PosEmb", "Wq", "Wk", "Wv", "Wo", "Wf1", "Wf2", "Wlm"}
	for idx, gs := range gradSlices {
		hasNonZero := false
		for _, v := range gs {
			if v != 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Fatalf("gradients for %s are all zero", names[idx])
		}
	}
}

func TestTensorAdamReducesLoss(t *testing.T) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))
	m := NewTensorModel(cfg, rng)
	c := NewTensorCache(cfg)
	g := NewTensorGrads(cfg)

	// Set up a batch.
	c.SeqLens[0] = 4
	c.SeqLens[1] = 3
	T := cfg.BlockSize
	seqs := [][]int{
		{26, 0, 1, 2},
		{26, 3, 4},
	}
	for b, seq := range seqs {
		for i, tok := range seq {
			c.Tokens[b*T+i] = tok
		}
	}

	// Forward pass 1.
	loss1 := Forward(m, c)

	// Backward.
	g.Zero()
	Backward(m, c, g)

	// Adam step.
	paramSlices := m.ParamSlices()
	gradSlices := g.GradSlices()
	adam := NewTensorAdam(paramSlices, 0.01, 0.85, 0.99, 1e-8)
	adam.Step(paramSlices, gradSlices, 1, 100)
	syncTransposedWeights(m)

	// Forward pass 2.
	loss2 := Forward(m, c)

	if loss2 >= loss1 {
		t.Fatalf("loss should decrease after Adam step: before=%f, after=%f", loss1, loss2)
	}
}

func TestTensorTrainConverges(t *testing.T) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))
	m := NewTensorModel(cfg, rng)

	docs := []string{"ab", "cd", "ef"}
	tokenizer := NewTokenizer(docs)

	// Override VocabSize to match the tokenizer.
	cfg.VocabSize = tokenizer.VocabSize()
	m = NewTensorModel(cfg, rng)

	opts := TrainOptions{
		NumSteps:     200,
		LearningRate: 0.005,
		Beta1:        0.85,
		Beta2:        0.99,
		EpsAdam:      1e-8,
	}

	var firstLoss, lastLoss float64
	err := TensorTrain(m, tokenizer, docs, opts, func(metrics StepMetrics) {
		if metrics.Step == 1 {
			firstLoss = metrics.Loss
		}
		lastLoss = metrics.Loss
	})
	if err != nil {
		t.Fatalf("TensorTrain failed: %v", err)
	}

	if lastLoss >= firstLoss {
		t.Fatalf("loss should decrease over training: first=%f, last=%f", firstLoss, lastLoss)
	}
}

func TestGenerateFastProducesOutput(t *testing.T) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))

	docs := []string{"ab", "cd", "ef"}
	tokenizer := NewTokenizer(docs)

	cfg.VocabSize = tokenizer.VocabSize()
	m := NewTensorModel(cfg, rng)

	// Train briefly so the model has seen some data.
	opts := TrainOptions{
		NumSteps:     50,
		LearningRate: 0.005,
		Beta1:        0.85,
		Beta2:        0.99,
		EpsAdam:      1e-8,
	}
	err := TensorTrain(m, tokenizer, docs, opts, nil)
	if err != nil {
		t.Fatalf("TensorTrain failed: %v", err)
	}

	sampleOpts := SampleOptions{
		NumSamples:  3,
		Temperature: 0.8,
	}
	samples := GenerateFast(m, tokenizer, sampleOpts, rng)

	if len(samples) != 3 {
		t.Fatalf("expected 3 samples, got %d", len(samples))
	}

	// At least one sample should be non-empty (the model may produce EOS immediately
	// for some samples, but not all with a trained model).
	hasNonEmpty := false
	for _, s := range samples {
		if len(s) > 0 {
			hasNonEmpty = true
			break
		}
	}
	if !hasNonEmpty {
		t.Fatalf("all generated samples are empty: %v", samples)
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkForward(b *testing.B) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))
	m := NewTensorModel(cfg, rng)
	c := NewTensorCache(cfg)

	// Fill batch with realistic data.
	T := cfg.BlockSize
	for bi := range cfg.Batch {
		c.SeqLens[bi] = 8
		for t := range 8 {
			c.Tokens[bi*T+t] = rng.Intn(cfg.VocabSize)
		}
	}

	b.ResetTimer()
	for range b.N {
		Forward(m, c)
	}
}

func BenchmarkForwardBackward(b *testing.B) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))
	m := NewTensorModel(cfg, rng)
	c := NewTensorCache(cfg)
	g := NewTensorGrads(cfg)

	T := cfg.BlockSize
	for bi := range cfg.Batch {
		c.SeqLens[bi] = 8
		for t := range 8 {
			c.Tokens[bi*T+t] = rng.Intn(cfg.VocabSize)
		}
	}

	b.ResetTimer()
	for range b.N {
		Forward(m, c)
		g.Zero()
		Backward(m, c, g)
	}
}

func BenchmarkTensorTrainStep(b *testing.B) {
	cfg := smallTestConfig()
	rng := rand.New(rand.NewSource(42))
	m := NewTensorModel(cfg, rng)
	c := NewTensorCache(cfg)
	g := NewTensorGrads(cfg)
	paramSlices := m.ParamSlices()
	gradSlices := g.GradSlices()
	adam := NewTensorAdam(paramSlices, 0.01, 0.85, 0.99, 1e-8)

	T := cfg.BlockSize
	for bi := range cfg.Batch {
		c.SeqLens[bi] = 8
		for t := range 8 {
			c.Tokens[bi*T+t] = rng.Intn(cfg.VocabSize)
		}
	}

	b.ResetTimer()
	for i := range b.N {
		Forward(m, c)
		g.Zero()
		Backward(m, c, g)
		adam.Step(paramSlices, gradSlices, i+1, b.N)
		syncTransposedWeights(m)
	}
}

package microgpt

import (
	"math"
	"math/rand"
	"testing"
)

func TestTrainingStepEndToEnd(t *testing.T) {
	docs := []string{"anna", "bob", "carla"}
	tok := NewTokenizer(docs)
	cfg := ModelConfig{
		NLayer:    1,
		NEmbd:     8,
		BlockSize: 8,
		NHead:     2,
	}

	rng := rand.New(rand.NewSource(42))
	model := NewModel(cfg, tok.VocabSize(), rng)
	adam := NewAdam(len(model.Params), 0.01, 0.85, 0.99, 1e-8)

	before := make([]float64, len(model.Params))
	for i, p := range model.Params {
		before[i] = p.Data
	}

	doc := docs[0]
	tokens := tok.EncodeWithBOS(doc)
	n := cfg.BlockSize
	if len(tokens)-1 < n {
		n = len(tokens) - 1
	}

	cache := NewKVCache(cfg.NLayer)
	loss := NewValue(0)
	for posID := 0; posID < n; posID++ {
		logits := model.ForwardToken(tokens[posID], posID, cache)
		probs := Softmax(logits)
		loss = loss.Add(probs[tokens[posID+1]].Log().Neg())
	}
	loss = loss.MulScalar(1.0 / float64(n))

	if math.IsNaN(loss.Data) || math.IsInf(loss.Data, 0) {
		t.Fatalf("loss is not finite: %v", loss.Data)
	}
	if loss.Data <= 0 {
		t.Fatalf("loss should be positive, got %v", loss.Data)
	}

	loss.Backward()
	adam.Step(model.Params, 0, 10)

	changed := false
	for i, p := range model.Params {
		if p.Data != before[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("expected at least one parameter update")
	}

	for i, p := range model.Params {
		if p.Grad != 0 {
			t.Fatalf("param %d grad = %v, want 0 after optimizer step", i, p.Grad)
		}
	}

	if got := len(cache.Keys[0]); got != n {
		t.Fatalf("cache key length = %d, want %d", got, n)
	}
	if got := len(cache.Values[0]); got != n {
		t.Fatalf("cache value length = %d, want %d", got, n)
	}
}

func TestTrainAPIUpdatesParametersAndReportsProgress(t *testing.T) {
	docs := []string{"anna", "bob", "carla"}
	tok := NewTokenizer(docs)
	cfg := ModelConfig{
		NLayer:    1,
		NEmbd:     8,
		BlockSize: 8,
		NHead:     2,
	}

	rng := rand.New(rand.NewSource(12))
	model := NewModel(cfg, tok.VocabSize(), rng)
	before := model.Params[0].Data

	opts := DefaultTrainOptions()
	opts.NumSteps = 3

	seen := 0
	err := Train(model, tok, docs, opts, func(metrics StepMetrics) {
		seen++
		if metrics.Total != opts.NumSteps {
			t.Fatalf("metrics.Total = %d, want %d", metrics.Total, opts.NumSteps)
		}
		if metrics.Step <= 0 || metrics.Step > opts.NumSteps {
			t.Fatalf("metrics.Step out of range: %d", metrics.Step)
		}
		if math.IsNaN(metrics.Loss) || math.IsInf(metrics.Loss, 0) {
			t.Fatalf("metrics.Loss not finite: %v", metrics.Loss)
		}
	})
	if err != nil {
		t.Fatalf("Train returned error: %v", err)
	}
	if seen != opts.NumSteps {
		t.Fatalf("progress callback count = %d, want %d", seen, opts.NumSteps)
	}
	if model.Params[0].Data == before {
		t.Fatal("expected parameter update from training")
	}
}

// BenchmarkAutogradTrainStep benchmarks the original autograd forward+backward+adam
// with the same dimensions as BenchmarkTensorTrainStep for direct comparison.
func BenchmarkAutogradTrainStep(b *testing.B) {
	docs := []string{"ab", "cd", "ef"}
	tok := NewTokenizer(docs)
	cfg := ModelConfig{
		NLayer:    1,
		NEmbd:     16,
		BlockSize: 16,
		NHead:     4,
	}
	rng := rand.New(rand.NewSource(42))
	model := NewModel(cfg, tok.VocabSize(), rng)
	adam := NewAdam(len(model.Params), 0.01, 0.85, 0.99, 1e-8)

	tokens := tok.EncodeWithBOS(docs[0])
	seqLen := len(tokens) - 1
	if seqLen > cfg.BlockSize {
		seqLen = cfg.BlockSize
	}

	b.ResetTimer()
	for i := range b.N {
		cache := NewKVCache(cfg.NLayer)
		loss := NewValue(0)
		for pos := range seqLen {
			logits := model.ForwardToken(tokens[pos], pos, cache)
			probs := Softmax(logits)
			loss = loss.Add(probs[tokens[pos+1]].Log().Neg())
		}
		loss = loss.MulScalar(1.0 / float64(seqLen))
		loss.Backward()
		adam.Step(model.Params, i, b.N)
	}
}

func TestGenerateSamplesAPI(t *testing.T) {
	docs := []string{"anna", "bob"}
	tok := NewTokenizer(docs)
	cfg := ModelConfig{
		NLayer:    1,
		NEmbd:     8,
		BlockSize: 8,
		NHead:     2,
	}

	rng := rand.New(rand.NewSource(4))
	model := NewModel(cfg, tok.VocabSize(), rng)

	opts := DefaultSampleOptions()
	opts.NumSamples = 4
	opts.Temperature = 0.7

	samples, err := GenerateSamples(model, tok, opts, rng)
	if err != nil {
		t.Fatalf("GenerateSamples returned error: %v", err)
	}
	if len(samples) != opts.NumSamples {
		t.Fatalf("len(samples) = %d, want %d", len(samples), opts.NumSamples)
	}
	for i, s := range samples {
		if len([]rune(s)) > cfg.BlockSize {
			t.Fatalf("sample %d length=%d exceeds block size %d", i, len([]rune(s)), cfg.BlockSize)
		}
	}
}

package microgpt

import (
	"math"
	"math/rand"
	"testing"
)

func TestSoftmaxNormalizes(t *testing.T) {
	logits := Vec{NewValue(1000), NewValue(1001), NewValue(1002)}
	probs := Softmax(logits)

	sum := 0.0
	for _, p := range probs {
		sum += p.Data
	}
	if !almostEqual(sum, 1.0, 1e-9) {
		t.Fatalf("softmax sum = %.12f, want 1", sum)
	}
	if !(probs[2].Data > probs[1].Data && probs[1].Data > probs[0].Data) {
		t.Fatalf("softmax should be monotonic with logits")
	}
}

func TestRMSNormScalesByRMS(t *testing.T) {
	x := Vec{NewValue(3), NewValue(4)}
	y := RMSNorm(x)

	den := math.Sqrt((3*3+4*4)/2.0 + 1e-5)
	want0 := 3 / den
	want1 := 4 / den

	if !almostEqual(y[0].Data, want0, 1e-9) {
		t.Fatalf("y[0] = %v, want %v", y[0].Data, want0)
	}
	if !almostEqual(y[1].Data, want1, 1e-9) {
		t.Fatalf("y[1] = %v, want %v", y[1].Data, want1)
	}
}

func TestNewModelParamCount(t *testing.T) {
	cfg := ModelConfig{
		NLayer:    1,
		NEmbd:     4,
		BlockSize: 3,
		NHead:     2,
	}
	vocab := 5
	rng := rand.New(rand.NewSource(7))
	m := NewModel(cfg, vocab, rng)

	// wte + wpe + lm_head + per-layer matrices
	want := vocab*cfg.NEmbd + cfg.BlockSize*cfg.NEmbd + vocab*cfg.NEmbd +
		4*cfg.NEmbd*cfg.NEmbd + 4*cfg.NEmbd*cfg.NEmbd + 4*cfg.NEmbd*cfg.NEmbd
	if got := len(m.Params); got != want {
		t.Fatalf("len(params) = %d, want %d", got, want)
	}
}

func TestForwardTokenOutputAndCacheGrowth(t *testing.T) {
	cfg := ModelConfig{
		NLayer:    1,
		NEmbd:     4,
		BlockSize: 4,
		NHead:     2,
	}
	vocab := 6
	rng := rand.New(rand.NewSource(123))
	m := NewModel(cfg, vocab, rng)
	cache := NewKVCache(cfg.NLayer)

	logits0 := m.ForwardToken(0, 0, cache)
	if len(logits0) != vocab {
		t.Fatalf("len(logits0) = %d, want %d", len(logits0), vocab)
	}
	if got := len(cache.Keys[0]); got != 1 {
		t.Fatalf("cache keys len after first token = %d, want 1", got)
	}
	if got := len(cache.Values[0]); got != 1 {
		t.Fatalf("cache values len after first token = %d, want 1", got)
	}
	if got := len(cache.Keys[0][0]); got != cfg.NEmbd {
		t.Fatalf("key vector len = %d, want %d", got, cfg.NEmbd)
	}

	_ = m.ForwardToken(1, 1, cache)
	if got := len(cache.Keys[0]); got != 2 {
		t.Fatalf("cache keys len after second token = %d, want 2", got)
	}
	if got := len(cache.Values[0]); got != 2 {
		t.Fatalf("cache values len after second token = %d, want 2", got)
	}
}

func TestNewModelPanicsOnInvalidHeadDim(t *testing.T) {
	cfg := ModelConfig{
		NLayer:    1,
		NEmbd:     5,
		BlockSize: 4,
		NHead:     2,
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when NEmbd is not divisible by NHead")
		}
	}()
	_ = NewModel(cfg, 6, rand.New(rand.NewSource(1)))
}

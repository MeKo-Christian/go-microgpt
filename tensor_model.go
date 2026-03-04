package microgpt

import (
	"math"
	"math/rand"
)

// TensorConfig holds model hyperparameters for the tensor implementation.
type TensorConfig struct {
	DModel    int // embedding dim (default 64)
	NHeads    int // attention heads (default 4)
	DFF       int // FFN hidden dim (= 4 * DModel)
	VocabSize int // 27 for names dataset
	BlockSize int // max sequence length (16)
	Batch     int // training batch size (16)
}

func (c TensorConfig) HeadDim() int { return c.DModel / c.NHeads }

// TensorModel holds all model parameters as flat float32 slices (row-major).
// Weight matrices are stored as W[outDim][inDim].
type TensorModel struct {
	Cfg TensorConfig

	TokEmb []float32 // [VocabSize][DModel]
	PosEmb []float32 // [BlockSize][DModel]

	Wq []float32 // [DModel][DModel]
	Wk []float32 // [DModel][DModel]
	Wv []float32 // [DModel][DModel]
	Wo []float32 // [DModel][DModel]

	Wf1 []float32 // [DFF][DModel]
	Wf2 []float32 // [DModel][DFF]

	Wlm []float32 // [VocabSize][DModel]

	// Transposed copies for efficient backward passes.
	WqT  []float32 // [DModel][DModel]
	WkT  []float32 // [DModel][DModel]
	WvT  []float32 // [DModel][DModel]
	WoT  []float32 // [DModel][DModel]
	Wf1T []float32 // [DModel][DFF]
	Wf2T []float32 // [DFF][DModel]
}

// TensorCache holds all intermediate activation buffers for a forward pass.
type TensorCache struct {
	RawEmb    []float32 // [Batch][BlockSize][DModel]
	Emb       []float32 // [Batch][BlockSize][DModel]
	EmbRMS    []float32 // [Batch][BlockSize]
	Norm1     []float32 // [Batch][BlockSize][DModel]
	Norm1RMS  []float32 // [Batch][BlockSize]
	Q         []float32 // [Batch][BlockSize][DModel]
	K         []float32 // [Batch][BlockSize][DModel]
	V         []float32 // [Batch][BlockSize][DModel]
	AttnScores []float32 // [Batch][NHeads][BlockSize][BlockSize]
	AttnOut   []float32 // [Batch][BlockSize][DModel]
	Res1      []float32 // [Batch][BlockSize][DModel]
	Norm2     []float32 // [Batch][BlockSize][DModel]
	Norm2RMS  []float32 // [Batch][BlockSize]
	FfPreReLU []float32 // [Batch][BlockSize][DFF]
	FfHidden  []float32 // [Batch][BlockSize][DFF]
	Res2      []float32 // [Batch][BlockSize][DModel]
	Logits    []float32 // [Batch][BlockSize][VocabSize]
	Probs     []float32 // [Batch][BlockSize][VocabSize]
	SeqLens   []int     // [Batch]
	Tokens    []int     // [Batch][BlockSize]
}

// TensorGrads holds gradient buffers mirroring TensorModel (excluding transposed copies).
type TensorGrads struct {
	TokEmb []float32 // [VocabSize][DModel]
	PosEmb []float32 // [BlockSize][DModel]

	Wq []float32 // [DModel][DModel]
	Wk []float32 // [DModel][DModel]
	Wv []float32 // [DModel][DModel]
	Wo []float32 // [DModel][DModel]

	Wf1 []float32 // [DFF][DModel]
	Wf2 []float32 // [DModel][DFF]

	Wlm []float32 // [VocabSize][DModel]
}

// NewTensorModel allocates and initialises a TensorModel with Kaiming/He init.
// Each weight element is drawn from N(0, 1/sqrt(fan_in)).
func NewTensorModel(cfg TensorConfig, rng *rand.Rand) *TensorModel {
	d := cfg.DModel
	dff := cfg.DFF
	v := cfg.VocabSize
	b := cfg.BlockSize

	initSlice := func(size int, fanIn int) []float32 {
		scale := float32(1.0 / math.Sqrt(float64(fanIn)))
		s := make([]float32, size)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * scale
		}
		return s
	}

	m := &TensorModel{
		Cfg: cfg,

		TokEmb: initSlice(v*d, d),
		PosEmb: initSlice(b*d, d),

		Wq: initSlice(d*d, d),
		Wk: initSlice(d*d, d),
		Wv: initSlice(d*d, d),
		Wo: initSlice(d*d, d),

		Wf1: initSlice(dff*d, d),
		Wf2: initSlice(d*dff, dff),

		Wlm: initSlice(v*d, d),

		// Transposed copies (will be filled by syncTransposedWeights).
		WqT:  make([]float32, d*d),
		WkT:  make([]float32, d*d),
		WvT:  make([]float32, d*d),
		WoT:  make([]float32, d*d),
		Wf1T: make([]float32, d*dff),
		Wf2T: make([]float32, dff*d),
	}

	syncTransposedWeights(m)
	return m
}

// NewTensorCache allocates all activation buffers for the given config.
func NewTensorCache(cfg TensorConfig) *TensorCache {
	B := cfg.Batch
	T := cfg.BlockSize
	D := cfg.DModel
	H := cfg.NHeads
	DFF := cfg.DFF
	V := cfg.VocabSize

	return &TensorCache{
		RawEmb:     make([]float32, B*T*D),
		Emb:        make([]float32, B*T*D),
		EmbRMS:     make([]float32, B*T),
		Norm1:      make([]float32, B*T*D),
		Norm1RMS:   make([]float32, B*T),
		Q:          make([]float32, B*T*D),
		K:          make([]float32, B*T*D),
		V:          make([]float32, B*T*D),
		AttnScores: make([]float32, B*H*T*T),
		AttnOut:    make([]float32, B*T*D),
		Res1:       make([]float32, B*T*D),
		Norm2:      make([]float32, B*T*D),
		Norm2RMS:   make([]float32, B*T),
		FfPreReLU:  make([]float32, B*T*DFF),
		FfHidden:   make([]float32, B*T*DFF),
		Res2:       make([]float32, B*T*D),
		Logits:     make([]float32, B*T*V),
		Probs:      make([]float32, B*T*V),
		SeqLens:    make([]int, B),
		Tokens:     make([]int, B*T),
	}
}

// NewTensorGrads allocates gradient buffers matching TensorModel parameters.
func NewTensorGrads(cfg TensorConfig) *TensorGrads {
	d := cfg.DModel
	dff := cfg.DFF
	v := cfg.VocabSize
	b := cfg.BlockSize

	return &TensorGrads{
		TokEmb: make([]float32, v*d),
		PosEmb: make([]float32, b*d),

		Wq: make([]float32, d*d),
		Wk: make([]float32, d*d),
		Wv: make([]float32, d*d),
		Wo: make([]float32, d*d),

		Wf1: make([]float32, dff*d),
		Wf2: make([]float32, d*dff),

		Wlm: make([]float32, v*d),
	}
}

// Zero resets all gradient buffers to zero.
func (g *TensorGrads) Zero() {
	for i := range g.TokEmb {
		g.TokEmb[i] = 0
	}
	for i := range g.PosEmb {
		g.PosEmb[i] = 0
	}
	for i := range g.Wq {
		g.Wq[i] = 0
	}
	for i := range g.Wk {
		g.Wk[i] = 0
	}
	for i := range g.Wv {
		g.Wv[i] = 0
	}
	for i := range g.Wo {
		g.Wo[i] = 0
	}
	for i := range g.Wf1 {
		g.Wf1[i] = 0
	}
	for i := range g.Wf2 {
		g.Wf2[i] = 0
	}
	for i := range g.Wlm {
		g.Wlm[i] = 0
	}
}

// syncTransposedWeights copies transposed versions of all weight matrices into
// the corresponding T-suffixed fields. For a matrix W[rows][cols], the transpose
// WT[cols][rows] satisfies WT[j*rows + i] = W[i*cols + j].
func syncTransposedWeights(m *TensorModel) {
	transpose := func(dst, src []float32, rows, cols int) {
		for i := range rows {
			for j := range cols {
				dst[j*rows+i] = src[i*cols+j]
			}
		}
	}

	d := m.Cfg.DModel
	dff := m.Cfg.DFF

	transpose(m.WqT, m.Wq, d, d)
	transpose(m.WkT, m.Wk, d, d)
	transpose(m.WvT, m.Wv, d, d)
	transpose(m.WoT, m.Wo, d, d)
	transpose(m.Wf1T, m.Wf1, dff, d)  // Wf1 is [DFF][DModel] -> Wf1T is [DModel][DFF]
	transpose(m.Wf2T, m.Wf2, d, dff)  // Wf2 is [DModel][DFF] -> Wf2T is [DFF][DModel]
}

// AllParams returns a single concatenated slice containing all parameter values.
func (m *TensorModel) AllParams() []float32 {
	slices := m.ParamSlices()
	total := 0
	for _, s := range slices {
		total += len(s)
	}
	all := make([]float32, 0, total)
	for _, s := range slices {
		all = append(all, s...)
	}
	return all
}

// ParamSlices returns a slice of all parameter slices in a canonical order.
func (m *TensorModel) ParamSlices() [][]float32 {
	return [][]float32{
		m.TokEmb,
		m.PosEmb,
		m.Wq,
		m.Wk,
		m.Wv,
		m.Wo,
		m.Wf1,
		m.Wf2,
		m.Wlm,
	}
}

// GradSlices returns a slice of all gradient slices in the same order as ParamSlices.
func (g *TensorGrads) GradSlices() [][]float32 {
	return [][]float32{
		g.TokEmb,
		g.PosEmb,
		g.Wq,
		g.Wk,
		g.Wv,
		g.Wo,
		g.Wf1,
		g.Wf2,
		g.Wlm,
	}
}

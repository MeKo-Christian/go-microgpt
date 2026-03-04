package microgpt

import (
	"math"
	"math/rand"
)

type Vec []*Value
type Matrix []Vec

type ModelConfig struct {
	NLayer    int
	NEmbd     int
	BlockSize int
	NHead     int
}

func (c ModelConfig) HeadDim() int {
	return c.NEmbd / c.NHead
}

type LayerParams struct {
	AttnWQ Matrix
	AttnWK Matrix
	AttnWV Matrix
	AttnWO Matrix
	MLPFC1 Matrix
	MLPFC2 Matrix
}

type Model struct {
	cfg    ModelConfig
	WTE    Matrix
	WPE    Matrix
	LMHead Matrix
	Layers []LayerParams
	Params []*Value
}

type KVCache struct {
	Keys   [][]Vec
	Values [][]Vec
}

func NewKVCache(nLayer int) *KVCache {
	return &KVCache{
		Keys:   make([][]Vec, nLayer),
		Values: make([][]Vec, nLayer),
	}
}

func NewModel(cfg ModelConfig, vocabSize int, rng *rand.Rand) *Model {
	if cfg.NEmbd%cfg.NHead != 0 {
		panic("NEmbd must be divisible by NHead")
	}

	matrix := func(nOut, nIn int, std float64) Matrix {
		m := make(Matrix, nOut)
		for i := range nOut {
			row := make(Vec, nIn)
			for j := range nIn {
				row[j] = NewValue(rng.NormFloat64() * std)
			}
			m[i] = row
		}
		return m
	}

	m := &Model{
		cfg:    cfg,
		WTE:    matrix(vocabSize, cfg.NEmbd, 0.08),
		WPE:    matrix(cfg.BlockSize, cfg.NEmbd, 0.08),
		LMHead: matrix(vocabSize, cfg.NEmbd, 0.08),
		Layers: make([]LayerParams, cfg.NLayer),
	}

	for li := 0; li < cfg.NLayer; li++ {
		m.Layers[li] = LayerParams{
			AttnWQ: matrix(cfg.NEmbd, cfg.NEmbd, 0.08),
			AttnWK: matrix(cfg.NEmbd, cfg.NEmbd, 0.08),
			AttnWV: matrix(cfg.NEmbd, cfg.NEmbd, 0.08),
			AttnWO: matrix(cfg.NEmbd, cfg.NEmbd, 0.08),
			MLPFC1: matrix(4*cfg.NEmbd, cfg.NEmbd, 0.08),
			MLPFC2: matrix(cfg.NEmbd, 4*cfg.NEmbd, 0.08),
		}
	}

	m.collectParams()
	return m
}

func (m *Model) Config() ModelConfig {
	return m.cfg
}

func (m *Model) collectParams() {
	params := make([]*Value, 0, 1<<16)
	appendMatrix := func(mat Matrix) {
		for _, row := range mat {
			params = append(params, row...)
		}
	}

	appendMatrix(m.WTE)
	appendMatrix(m.WPE)
	appendMatrix(m.LMHead)
	for _, layer := range m.Layers {
		appendMatrix(layer.AttnWQ)
		appendMatrix(layer.AttnWK)
		appendMatrix(layer.AttnWV)
		appendMatrix(layer.AttnWO)
		appendMatrix(layer.MLPFC1)
		appendMatrix(layer.MLPFC2)
	}
	m.Params = params
}

func Linear(x Vec, w Matrix) Vec {
	out := make(Vec, len(w))
	for i, row := range w {
		sum := NewValue(0)
		for j, wij := range row {
			sum = sum.Add(wij.Mul(x[j]))
		}
		out[i] = sum
	}
	return out
}

func Softmax(logits Vec) Vec {
	maxVal := logits[0].Data
	for _, l := range logits[1:] {
		if l.Data > maxVal {
			maxVal = l.Data
		}
	}

	exps := make(Vec, len(logits))
	total := NewValue(0)
	for i, l := range logits {
		e := l.AddScalar(-maxVal).Exp()
		exps[i] = e
		total = total.Add(e)
	}

	probs := make(Vec, len(logits))
	for i, e := range exps {
		probs[i] = e.Div(total)
	}
	return probs
}

func RMSNorm(x Vec) Vec {
	ms := NewValue(0)
	for _, xi := range x {
		ms = ms.Add(xi.Mul(xi))
	}
	ms = ms.MulScalar(1.0 / float64(len(x)))
	scale := ms.AddScalar(1e-5).Pow(-0.5)

	out := make(Vec, len(x))
	for i, xi := range x {
		out[i] = xi.Mul(scale)
	}
	return out
}

func (m *Model) ForwardToken(tokenID, posID int, cache *KVCache) Vec {
	headDim := m.cfg.HeadDim()

	tokEmb := m.WTE[tokenID]
	posEmb := m.WPE[posID]
	x := make(Vec, m.cfg.NEmbd)
	for i := range x {
		x[i] = tokEmb[i].Add(posEmb[i])
	}
	x = RMSNorm(x)

	for li := 0; li < m.cfg.NLayer; li++ {
		layer := m.Layers[li]

		xResidual := x
		x = RMSNorm(x)
		q := Linear(x, layer.AttnWQ)
		k := Linear(x, layer.AttnWK)
		v := Linear(x, layer.AttnWV)
		cache.Keys[li] = append(cache.Keys[li], k)
		cache.Values[li] = append(cache.Values[li], v)

		xAttn := make(Vec, 0, m.cfg.NEmbd)
		for h := 0; h < m.cfg.NHead; h++ {
			hs := h * headDim
			qh := q[hs : hs+headDim]

			attnLogits := make(Vec, len(cache.Keys[li]))
			for t := range cache.Keys[li] {
				kh := cache.Keys[li][t][hs : hs+headDim]
				logit := NewValue(0)
				for j := 0; j < headDim; j++ {
					logit = logit.Add(qh[j].Mul(kh[j]))
				}
				attnLogits[t] = logit.MulScalar(1.0 / math.Sqrt(float64(headDim)))
			}

			attnWeights := Softmax(attnLogits)
			headOut := make(Vec, headDim)
			for j := 0; j < headDim; j++ {
				sum := NewValue(0)
				for t := range cache.Values[li] {
					vh := cache.Values[li][t][hs : hs+headDim]
					sum = sum.Add(attnWeights[t].Mul(vh[j]))
				}
				headOut[j] = sum
			}
			xAttn = append(xAttn, headOut...)
		}

		x = Linear(xAttn, layer.AttnWO)
		for i := range x {
			x[i] = x[i].Add(xResidual[i])
		}

		xResidual = x
		x = RMSNorm(x)
		x = Linear(x, layer.MLPFC1)
		for i := range x {
			x[i] = x[i].ReLU()
		}
		x = Linear(x, layer.MLPFC2)
		for i := range x {
			x[i] = x[i].Add(xResidual[i])
		}
	}

	return Linear(x, m.LMHead)
}

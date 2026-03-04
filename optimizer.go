package microgpt

import "math"

type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Eps   float64
	M     []float64
	V     []float64
}

func NewAdam(numParams int, lr, beta1, beta2, eps float64) *Adam {
	return &Adam{
		LR:    lr,
		Beta1: beta1,
		Beta2: beta2,
		Eps:   eps,
		M:     make([]float64, numParams),
		V:     make([]float64, numParams),
	}
}

func (a *Adam) Step(params []*Value, step, totalSteps int) {
	lrT := a.LR * (1 - float64(step)/float64(totalSteps))
	for i, p := range params {
		g := p.Grad
		a.M[i] = a.Beta1*a.M[i] + (1-a.Beta1)*g
		a.V[i] = a.Beta2*a.V[i] + (1-a.Beta2)*g*g

		mHat := a.M[i] / (1 - math.Pow(a.Beta1, float64(step+1)))
		vHat := a.V[i] / (1 - math.Pow(a.Beta2, float64(step+1)))

		p.Data -= lrT * mHat / (math.Sqrt(vHat) + a.Eps)
		p.Grad = 0
	}
}

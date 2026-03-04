package microgpt

import "math"

// Value is a scalar in a computation graph for reverse-mode autodiff.
type Value struct {
	Data       float64
	Grad       float64
	children   []*Value
	localGrads []float64
}

func NewValue(data float64) *Value {
	return &Value{Data: data}
}

func (v *Value) Add(other *Value) *Value {
	return &Value{
		Data:       v.Data + other.Data,
		children:   []*Value{v, other},
		localGrads: []float64{1, 1},
	}
}

func (v *Value) AddScalar(c float64) *Value {
	return &Value{
		Data:       v.Data + c,
		children:   []*Value{v},
		localGrads: []float64{1},
	}
}

func (v *Value) Mul(other *Value) *Value {
	return &Value{
		Data:       v.Data * other.Data,
		children:   []*Value{v, other},
		localGrads: []float64{other.Data, v.Data},
	}
}

func (v *Value) MulScalar(c float64) *Value {
	return &Value{
		Data:       v.Data * c,
		children:   []*Value{v},
		localGrads: []float64{c},
	}
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

func (v *Value) DivScalar(c float64) *Value {
	return v.MulScalar(1.0 / c)
}

func (v *Value) Pow(power float64) *Value {
	return &Value{
		Data:       math.Pow(v.Data, power),
		children:   []*Value{v},
		localGrads: []float64{power * math.Pow(v.Data, power-1)},
	}
}

func (v *Value) Log() *Value {
	return &Value{
		Data:       math.Log(v.Data),
		children:   []*Value{v},
		localGrads: []float64{1 / v.Data},
	}
}

func (v *Value) Exp() *Value {
	ev := math.Exp(v.Data)
	return &Value{
		Data:       ev,
		children:   []*Value{v},
		localGrads: []float64{ev},
	}
}

func (v *Value) ReLU() *Value {
	grad := 0.0
	if v.Data > 0 {
		grad = 1.0
	}
	data := 0.0
	if v.Data > 0 {
		data = v.Data
	}
	return &Value{
		Data:       data,
		children:   []*Value{v},
		localGrads: []float64{grad},
	}
}

func (v *Value) Neg() *Value {
	return v.MulScalar(-1)
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

func (v *Value) Backward() {
	topo := make([]*Value, 0, 1024)
	visited := make(map[*Value]bool, 1024)

	var buildTopo func(node *Value)
	buildTopo = func(node *Value) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, child := range node.children {
			buildTopo(child)
		}
		topo = append(topo, node)
	}

	buildTopo(v)
	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		for ci, child := range node.children {
			child.Grad += node.localGrads[ci] * node.Grad
		}
	}
}

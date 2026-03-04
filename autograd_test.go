package microgpt

import (
	"math"
	"testing"
)

func almostEqual(got, want, tol float64) bool {
	return math.Abs(got-want) <= tol
}

func TestValueBackwardAlgebra(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := NewValue(4.0)

	// f(a,b,c) = ((a*b)+c)^2 / a
	u := a.Mul(b).Add(c)
	f := u.Pow(2).Div(a)
	f.Backward()

	if !almostEqual(f.Data, 50.0, 1e-9) {
		t.Fatalf("f.Data = %v, want 50", f.Data)
	}
	if !almostEqual(a.Grad, 5.0, 1e-9) {
		t.Fatalf("a.Grad = %v, want 5", a.Grad)
	}
	if !almostEqual(b.Grad, 20.0, 1e-9) {
		t.Fatalf("b.Grad = %v, want 20", b.Grad)
	}
	if !almostEqual(c.Grad, 10.0, 1e-9) {
		t.Fatalf("c.Grad = %v, want 10", c.Grad)
	}
}

func TestValueBackwardSharedNode(t *testing.T) {
	x := NewValue(3.0)
	y := x.Mul(x).Add(x) // dy/dx = 2x + 1
	y.Backward()

	if !almostEqual(y.Data, 12.0, 1e-9) {
		t.Fatalf("y.Data = %v, want 12", y.Data)
	}
	if !almostEqual(x.Grad, 7.0, 1e-9) {
		t.Fatalf("x.Grad = %v, want 7", x.Grad)
	}
}

func TestValueNonlinearOps(t *testing.T) {
	x := NewValue(1.7)
	y := x.Exp().Log().ReLU() // y == x for x > 0
	y.Backward()

	if !almostEqual(y.Data, 1.7, 1e-9) {
		t.Fatalf("y.Data = %v, want 1.7", y.Data)
	}
	if !almostEqual(x.Grad, 1.0, 1e-9) {
		t.Fatalf("x.Grad = %v, want 1", x.Grad)
	}
}

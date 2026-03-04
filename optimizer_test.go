package microgpt

import (
	"math"
	"testing"
)

func TestAdamSingleStep(t *testing.T) {
	p := NewValue(1.0)
	p.Grad = 1.0
	params := []*Value{p}

	opt := NewAdam(1, 0.01, 0.9, 0.999, 1e-8)
	opt.Step(params, 0, 10)

	want := 1.0 - 0.01 // for first step with g=1, bias-corrected mhat=1, vhat=1
	if !almostEqual(p.Data, want, 1e-6) {
		t.Fatalf("p.Data = %.10f, want %.10f", p.Data, want)
	}
	if p.Grad != 0 {
		t.Fatalf("p.Grad = %v, want 0 after step", p.Grad)
	}
}

func TestAdamUsesLRDecay(t *testing.T) {
	p0 := NewValue(1.0)
	p0.Grad = 1.0
	p1 := NewValue(1.0)
	p1.Grad = 1.0

	opt0 := NewAdam(1, 0.01, 0.9, 0.999, 1e-8)
	opt1 := NewAdam(1, 0.01, 0.9, 0.999, 1e-8)

	opt0.Step([]*Value{p0}, 0, 10) // higher lr_t
	opt1.Step([]*Value{p1}, 9, 10) // lower lr_t

	delta0 := math.Abs(1.0 - p0.Data)
	delta1 := math.Abs(1.0 - p1.Data)
	if !(delta0 > delta1) {
		t.Fatalf("expected larger update at earlier step: delta0=%g delta1=%g", delta0, delta1)
	}
}

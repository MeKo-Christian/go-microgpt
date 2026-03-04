package microgpt

import (
	"math"
	"testing"
)

func TestDotF32Go(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	got := DotF32Go(a, b, 3)
	if math.Abs(float64(got-32)) > 1e-6 {
		t.Fatalf("DotF32Go: expected 32, got %f", got)
	}
}

func TestMatVecF32(t *testing.T) {
	// W is 3x2 (outDim=3, inDim=2), row-major:
	// [1 2]
	// [3 4]
	// [5 6]
	W := []float32{1, 2, 3, 4, 5, 6}
	x := []float32{7, 8}
	y := make([]float32, 3)
	MatVecF32(y, W, x, 3, 2)

	// y[0] = 1*7+2*8 = 23, y[1] = 3*7+4*8 = 53, y[2] = 5*7+6*8 = 83
	expected := []float32{23, 53, 83}
	for i := range 3 {
		if math.Abs(float64(y[i]-expected[i])) > 1e-6 {
			t.Fatalf("MatVecF32: y[%d] expected %f, got %f", i, expected[i], y[i])
		}
	}
}

func TestMatVec2F32(t *testing.T) {
	W := []float32{1, 2, 3, 4, 5, 6}
	x0 := []float32{7, 8}
	x1 := []float32{9, 10}
	y0 := make([]float32, 3)
	y1 := make([]float32, 3)
	MatVec2F32(y0, y1, W, x0, x1, 3, 2)

	// Verify against individual MatVecF32 calls.
	ref0 := make([]float32, 3)
	ref1 := make([]float32, 3)
	MatVecF32(ref0, W, x0, 3, 2)
	MatVecF32(ref1, W, x1, 3, 2)

	for i := range 3 {
		if math.Abs(float64(y0[i]-ref0[i])) > 1e-6 {
			t.Fatalf("MatVec2F32: y0[%d] expected %f, got %f", i, ref0[i], y0[i])
		}
		if math.Abs(float64(y1[i]-ref1[i])) > 1e-6 {
			t.Fatalf("MatVec2F32: y1[%d] expected %f, got %f", i, ref1[i], y1[i])
		}
	}
}

func TestMatVecReLUF32(t *testing.T) {
	// W is 2x3 (outDim=2, inDim=3):
	// [ 1  2  3]    -> dot with [1,-1,0] =  1*1 + 2*(-1) + 3*0 = -1 (negative, zeroed)
	// [-1 -2 -3]    -> dot with [1,-1,0] = -1*1 +(-2)*(-1)+(-3)*0 = 1 (positive, passes)
	W := []float32{1, 2, 3, -1, -2, -3}
	x := []float32{1, -1, 0}
	preRelu := make([]float32, 2)
	hidden := make([]float32, 2)
	MatVecReLUF32(preRelu, hidden, W, x, 2, 3)

	// preRelu should store the linear output before ReLU.
	if math.Abs(float64(preRelu[0]-(-1))) > 1e-6 {
		t.Fatalf("preRelu[0] expected -1, got %f", preRelu[0])
	}
	if math.Abs(float64(preRelu[1]-1)) > 1e-6 {
		t.Fatalf("preRelu[1] expected 1, got %f", preRelu[1])
	}

	// hidden should be ReLU applied: negative zeroed, positive passes.
	if hidden[0] != 0 {
		t.Fatalf("hidden[0] should be 0 (ReLU of -1), got %f", hidden[0])
	}
	if math.Abs(float64(hidden[1]-1)) > 1e-6 {
		t.Fatalf("hidden[1] should be 1, got %f", hidden[1])
	}
}

func TestMatVecResidualF32(t *testing.T) {
	W := []float32{1, 2, 3, 4}
	x := []float32{5, 6}
	base := []float32{10, 20}
	y := make([]float32, 2)
	MatVecResidualF32(y, base, W, x, 2, 2)

	// y[0] = 10 + (1*5+2*6) = 10+17 = 27
	// y[1] = 20 + (3*5+4*6) = 20+39 = 59
	expected := []float32{27, 59}
	for i := range 2 {
		if math.Abs(float64(y[i]-expected[i])) > 1e-6 {
			t.Fatalf("MatVecResidualF32: y[%d] expected %f, got %f", i, expected[i], y[i])
		}
	}
}

func TestRMSNormF32(t *testing.T) {
	x := []float32{3, 4}
	out := make([]float32, 2)
	rms := RMSNormF32(out, x, 2)

	// rms = sqrt((9+16)/2 + 1e-8) = sqrt(12.5 + 1e-8) ~ 3.5355339
	expectedRMS := float32(math.Sqrt(12.5 + 1e-8))
	if math.Abs(float64(rms-expectedRMS)) > 1e-4 {
		t.Fatalf("RMS value: expected %f, got %f", expectedRMS, rms)
	}

	// Verify output has unit RMS: sqrt(mean(out^2)) should be ~1.
	var sumSq float64
	for i := range 2 {
		sumSq += float64(out[i]) * float64(out[i])
	}
	outRMS := math.Sqrt(sumSq/2 + 1e-8)
	if math.Abs(outRMS-1.0) > 1e-3 {
		t.Fatalf("output RMS should be ~1.0, got %f", outRMS)
	}
}

func TestSoftmaxF32(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0}
	SoftmaxF32(x, 3)

	// Verify all positive.
	for i := range 3 {
		if x[i] <= 0 {
			t.Fatalf("softmax output[%d] should be positive, got %f", i, x[i])
		}
	}

	// Verify sum to 1.
	var sum float64
	for i := range 3 {
		sum += float64(x[i])
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Fatalf("softmax outputs should sum to 1, got %f", sum)
	}

	// Largest input (3.0 at index 2) maps to largest output.
	if x[2] <= x[1] || x[1] <= x[0] {
		t.Fatalf("softmax should preserve ordering: got %v", x)
	}
}

func TestFastExpF32(t *testing.T) {
	// The degree-3 minimax polynomial for 2^r on [0,1) has peak relative error
	// around 6e-3 when combined with float32 precision. Verify the full range
	// [-10, 10] stays within that bound.
	var maxRelErr float64
	for i := -100; i <= 100; i++ {
		x := float32(i) / 10.0
		got := float64(FastExpF32(x))
		want := math.Exp(float64(x))

		relErr := math.Abs(got-want) / math.Max(math.Abs(want), 1e-30)
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > 6e-3 {
			t.Fatalf("FastExpF32(%f): got %f, want %f, relErr %e", x, got, want, relErr)
		}
	}
	t.Logf("max relative error over [-10,10]: %e", maxRelErr)

	// Verify extreme values don't panic or return inf/nan.
	for _, x := range []float32{-10, 10, -50, 50} {
		v := FastExpF32(x)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("FastExpF32(%f) returned %f", x, v)
		}
	}
}

func TestSoftmaxFastF32(t *testing.T) {
	// Compare SoftmaxFastF32 against SoftmaxF32.
	xFast := []float32{1.0, 2.0, 3.0, -1.0, 0.5}
	xRef := make([]float32, len(xFast))
	copy(xRef, xFast)

	SoftmaxF32(xRef, len(xRef))
	SoftmaxFastF32(xFast, len(xFast))

	for i := range xFast {
		diff := math.Abs(float64(xFast[i] - xRef[i]))
		if diff > 1e-3 {
			t.Fatalf("SoftmaxFastF32 vs SoftmaxF32: index %d, fast=%f ref=%f diff=%e",
				i, xFast[i], xRef[i], diff)
		}
	}
}

func TestCrossEntropyF32(t *testing.T) {
	// Known probability: if probs[target] = 0.5, loss = -log(0.5) = ln(2) ~ 0.6931
	probs := []float32{0.2, 0.5, 0.3}
	loss := CrossEntropyF32(probs, 1)
	expected := float32(-math.Log(0.5))
	if math.Abs(float64(loss-expected)) > 1e-5 {
		t.Fatalf("CrossEntropyF32: expected %f, got %f", expected, loss)
	}

	// Verify floor for near-zero probability: should clamp to -log(1e-10).
	probsZero := []float32{0.0, 1.0}
	lossZero := CrossEntropyF32(probsZero, 0)
	expectedFloor := float32(-math.Log(1e-10))
	if math.Abs(float64(lossZero-expectedFloor)) > 1e-2 {
		t.Fatalf("CrossEntropyF32 floor: expected %f, got %f", expectedFloor, lossZero)
	}
}

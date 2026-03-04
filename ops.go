package microgpt

import "math"

// DotF32Go computes the dot product of two float32 slices.
func DotF32Go(a, b []float32, n int) float32 {
	var sum float32
	for i := range n {
		sum += a[i] * b[i]
	}
	return sum
}

// MatVecF32 computes y[j] = sum_i W[j*inDim+i] * x[i] for j in [0, outDim).
func MatVecF32(y, W, x []float32, outDim, inDim int) {
	for j := range outDim {
		y[j] = DotF32Go(W[j*inDim:], x, inDim)
	}
}

// MatVec2F32 computes y0 = W@x0 and y1 = W@x1 in one pass over W (halves weight bandwidth).
func MatVec2F32(y0, y1, W, x0, x1 []float32, outDim, inDim int) {
	for j := range outDim {
		row := W[j*inDim:]
		var s0, s1 float32
		for i := range inDim {
			w := row[i]
			s0 += w * x0[i]
			s1 += w * x1[i]
		}
		y0[j] = s0
		y1[j] = s1
	}
}

// MatVecReLUF32 computes y[j] = max(0, sum_i W[j*inDim+i]*x[i]) and stores pre-relu values.
// preRelu stores the linear output before ReLU (needed for backward pass).
func MatVecReLUF32(preRelu, hidden, W, x []float32, outDim, inDim int) {
	for j := range outDim {
		v := DotF32Go(W[j*inDim:], x, inDim)
		preRelu[j] = v
		if v > 0 {
			hidden[j] = v
		} else {
			hidden[j] = 0
		}
	}
}

// MatVecResidualF32 computes y[j] = base[j] + sum_i W[j*inDim+i]*x[i].
func MatVecResidualF32(y, base, W, x []float32, outDim, inDim int) {
	for j := range outDim {
		y[j] = base[j] + DotF32Go(W[j*inDim:], x, inDim)
	}
}

// RMSNormF32 computes RMS normalization: out[i] = x[i] / rms, returns the rms value.
// rms = sqrt(mean(x^2) + eps) where eps = 1e-8
func RMSNormF32(out, x []float32, n int) float32 {
	var sumSq float32
	for i := range n {
		sumSq += x[i] * x[i]
	}
	rms := float32(math.Sqrt(float64(sumSq/float32(n)) + 1e-8))
	invRms := 1.0 / rms
	for i := range n {
		out[i] = x[i] * invRms
	}
	return rms
}

// SoftmaxF32 computes softmax in-place over a float32 slice of length n.
// Uses the max-subtraction trick for numerical stability.
func SoftmaxF32(x []float32, n int) {
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sum float32
	for i := range n {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	invSum := 1.0 / sum
	for i := range n {
		x[i] *= invSum
	}
}

// FastExpF32 computes a fast approximation of exp(x) using a degree-3 polynomial.
// Relative error ~1e-4, acceptable for softmax.
func FastExpF32(x float32) float32 {
	// Convert to base-2: exp(x) = 2^(x * log2(e))
	x *= 1.442695041 // log2(e)

	// Clamp to avoid overflow/underflow in float32
	if x < -126 {
		x = -126
	} else if x > 126 {
		x = 126
	}

	// Split into integer and fractional parts
	xi := float32(math.Floor(float64(x)))
	r := x - xi

	// Construct 2^xi via IEEE 754 bit manipulation
	pow2i := math.Float32frombits(uint32(int32(xi)+127) << 23)

	// Degree-3 minimax polynomial for 2^r on [0, 1)
	p := ((0.0558011*r+0.2402265)*r+0.6931472)*r + 1.0

	return pow2i * p
}

// SoftmaxFastF32 computes softmax using FastExpF32 instead of math.Exp.
func SoftmaxFastF32(x []float32, n int) {
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sum float32
	for i := range n {
		x[i] = FastExpF32(x[i] - maxVal)
		sum += x[i]
	}
	invSum := 1.0 / sum
	for i := range n {
		x[i] *= invSum
	}
}

// VecAddF32 computes dst[i] += src[i] for i in [0, n).
func VecAddF32(dst, src []float32, n int) {
	for i := range n {
		dst[i] += src[i]
	}
}

// VecScaleF32 computes dst[i] *= s for i in [0, n).
func VecScaleF32(dst []float32, s float32, n int) {
	for i := range n {
		dst[i] *= s
	}
}

// VecZeroF32 zeros a float32 slice.
func VecZeroF32(dst []float32, n int) {
	for i := range n {
		dst[i] = 0
	}
}

// CrossEntropyF32 computes -log(probs[target]) for cross-entropy loss.
func CrossEntropyF32(probs []float32, target int) float32 {
	p := probs[target]
	if p < 1e-10 {
		p = 1e-10
	}
	return -float32(math.Log(float64(p)))
}

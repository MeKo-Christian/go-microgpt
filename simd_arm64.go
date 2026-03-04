//go:build arm64

package microgpt

// DotF32 computes the dot product of a[0:n] and b[0:n] using NEON.
//
//go:noescape
func DotF32(a, b *float32, n int) float32

// VecAddF32SIMD computes dst[i] += src[i] for i in [0, n) using NEON.
//
//go:noescape
func VecAddF32SIMD(dst, src *float32, n int)

// VecScaleF32SIMD computes dst[i] *= s for i in [0, n) using NEON.
//
//go:noescape
func VecScaleF32SIMD(dst *float32, s float32, n int)

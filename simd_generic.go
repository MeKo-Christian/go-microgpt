//go:build !amd64 && !arm64

package microgpt

import "unsafe"

// DotF32 computes the dot product using pure Go (fallback for non-SIMD platforms).
func DotF32(a, b *float32, n int) float32 {
	as := unsafe.Slice(a, n)
	bs := unsafe.Slice(b, n)
	return DotF32Go(as, bs, n)
}

// VecAddF32SIMD is a pure Go fallback for vector addition.
func VecAddF32SIMD(dst, src *float32, n int) {
	d := unsafe.Slice(dst, n)
	s := unsafe.Slice(src, n)
	VecAddF32(d, s, n)
}

// VecScaleF32SIMD is a pure Go fallback for vector scaling.
func VecScaleF32SIMD(dst *float32, s float32, n int) {
	d := unsafe.Slice(dst, n)
	VecScaleF32(d, s, n)
}

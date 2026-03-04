#include "textflag.h"

// func DotF32(a, b *float32, n int) float32
// ABI0 layout: a+0(FP)=8, b+8(FP)=8, n+16(FP)=8, ret+24(FP)=4  total=28
TEXT ·DotF32(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2

	// Zero accumulators V0, V1
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16

dot_loop8:
	CMP  $8, R2
	BLT  dot_loop4
	// Load 8 floats (2x4) from each array
	VLD1 (R0), [V2.S4, V3.S4]
	ADD  $32, R0
	VLD1 (R1), [V4.S4, V5.S4]
	ADD  $32, R1
	// acc += a * b  using VFMLA (Vd += Vn * Vm)
	VFMLA V4.S4, V2.S4, V0.S4
	VFMLA V5.S4, V3.S4, V1.S4
	SUB  $8, R2
	B    dot_loop8

dot_loop4:
	CMP  $4, R2
	BLT  dot_reduce
	VLD1 (R0), [V2.S4]
	ADD  $16, R0
	VLD1 (R1), [V3.S4]
	ADD  $16, R1
	VFMLA V3.S4, V2.S4, V0.S4
	SUB  $4, R2
	B    dot_loop4

dot_reduce:
	// Combine two accumulators
	VADD V0.S4, V1.S4, V0.S4
	// Horizontal sum across all four lanes
	VADDV V0.S4, V0
	// V0.S[0] / F0 now holds the vector sum

	CBZ  R2, dot_done

	// Scalar tail
dot_scalar:
	FMOVS (R0), F2
	FMOVS (R1), F3
	FMADDS F3, F0, F2, F0
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, dot_scalar

dot_done:
	FMOVS F0, ret+24(FP)
	RET

// func VecAddF32SIMD(dst, src *float32, n int)
// ABI0 layout: dst+0(FP)=8, src+8(FP)=8, n+16(FP)=8  total=24
TEXT ·VecAddF32SIMD(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2

vadd_loop4:
	CMP  $4, R2
	BLT  vadd_tail
	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	VADD V0.S4, V1.S4, V0.S4
	VST1 [V0.S4], (R0)
	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    vadd_loop4

vadd_tail:
	CBZ  R2, vadd_done

vadd_scalar:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FADDS F0, F1, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, vadd_scalar

vadd_done:
	RET

// func VecScaleF32SIMD(dst *float32, s float32, n int)
// ABI0 layout: dst+0(FP)=8, s+8(FP)=4, pad=4, n+16(FP)=8  total=24
TEXT ·VecScaleF32SIMD(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	FMOVS s+8(FP), F1
	MOVD n+16(FP), R2
	// Broadcast scalar F1 to all 4 lanes of V1
	VDUP V1.S[0], V1.S4

vscale_loop4:
	CMP  $4, R2
	BLT  vscale_tail
	VLD1 (R0), [V0.S4]
	// V2 = V0 * V1 via VFMLA: zero V2, then V2 += V0 * V1
	VEOR V2.B16, V2.B16, V2.B16
	VFMLA V1.S4, V0.S4, V2.S4
	VST1 [V2.S4], (R0)
	ADD  $16, R0
	SUB  $4, R2
	B    vscale_loop4

vscale_tail:
	CBZ  R2, vscale_done

vscale_scalar:
	FMOVS (R0), F0
	FMULS F0, F1, F0
	FMOVS F0, (R0)
	ADD  $4, R0
	SUB  $1, R2
	CBNZ R2, vscale_scalar

vscale_done:
	RET

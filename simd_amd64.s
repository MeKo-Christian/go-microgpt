#include "textflag.h"

// func DotF32(a, b *float32, n int) float32
// ABI0 layout: a+0(FP)=8, b+8(FP)=8, n+16(FP)=8, ret+24(FP)=4  total=28
TEXT ·DotF32(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX
	VXORPS Y0, Y0, Y0          // acc0 = 0
	VXORPS Y1, Y1, Y1          // acc1 = 0 (2-way unroll)

loop16:
	CMPQ CX, $16
	JL loop8
	VMOVUPS   (SI), Y2
	VMOVUPS 32(SI), Y3
	VFMADD231PS   (DI), Y2, Y0
	VFMADD231PS 32(DI), Y3, Y1
	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP loop16

loop8:
	CMPQ CX, $8
	JL tail
	VMOVUPS (SI), Y2
	VFMADD231PS (DI), Y2, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX

tail:
	VADDPS Y0, Y1, Y0          // combine accumulators
	// Horizontal sum of Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X0, X1, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	// Scalar tail for remaining < 8 elements
	TESTQ CX, CX
	JZ done

tail_loop:
	VMOVSS (SI), X2
	VMOVSS (DI), X3
	VFMADD231SS X3, X2, X0
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JNZ tail_loop

done:
	VMOVSS X0, ret+24(FP)
	VZEROUPPER
	RET

// func VecAddF32SIMD(dst, src *float32, n int)
// ABI0 layout: dst+0(FP)=8, src+8(FP)=8, n+16(FP)=8  total=24
TEXT ·VecAddF32SIMD(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX

add_loop8:
	CMPQ CX, $8
	JL add_tail
	VMOVUPS (DI), Y0
	VADDPS (SI), Y0, Y0
	VMOVUPS Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $8, CX
	JMP add_loop8

add_tail:
	TESTQ CX, CX
	JZ add_done

add_tail_loop:
	VMOVSS (DI), X0
	VADDSS (SI), X0, X0
	VMOVSS X0, (DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ CX
	JNZ add_tail_loop

add_done:
	VZEROUPPER
	RET

// func VecScaleF32SIMD(dst *float32, s float32, n int)
// ABI0 layout: dst+0(FP)=8, s+8(FP)=4, pad=4, n+16(FP)=8  total=24
TEXT ·VecScaleF32SIMD(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	VBROADCASTSS s+8(FP), Y1
	MOVQ n+16(FP), CX

scale_loop8:
	CMPQ CX, $8
	JL scale_tail
	VMOVUPS (DI), Y0
	VMULPS Y1, Y0, Y0
	VMOVUPS Y0, (DI)
	ADDQ $32, DI
	SUBQ $8, CX
	JMP scale_loop8

scale_tail:
	TESTQ CX, CX
	JZ scale_done

scale_tail_loop:
	VMOVSS (DI), X0
	VMULSS X1, X0, X0
	VMOVSS X0, (DI)
	ADDQ $4, DI
	DECQ CX
	JNZ scale_tail_loop

scale_done:
	VZEROUPPER
	RET

#ifndef MATMUL_H
#define MATMUL_H
#include <stdio.h>
#include <string.h>
#include "libpopcnt.h"
#define MC  256
#define KC  64
#define NC  256

#define MR  4
#define NR  4
#define ENCODE_BIT 32
#define MASK(a) ( (a) + ( -(a) & -((0)>(a)) ) )
const uint32_t UBIT = ~0;
//
//  Local buffers for storing panels from A, B and C
//
static uint32_t _A[MC*KC];
static uint32_t _B[KC*NC];

static inline uint32_t popcnt32(uint32_t x)
{
  __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
  return x;
}
//
//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk(int k, uint32_t *A, int incRowA, int incColA, uint32_t *buffer){
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A(int mc, int kc, uint32_t *A, int incRowA, int incColA, uint32_t *buffer){
    int i, j, mp  = mc / MR, _mr = mc % MR;

    for (i=0; i<mp; ++i) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR; ++i) {
                buffer[i] = UBIT;
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, uint32_t *B, int incRowB, int incColB, uint32_t *buffer){
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

//
//  Packing panels from B with padding if required
//
static void
pack_B(int kc, int nc, uint32_t *B, int incRowB, int incColB, uint32_t *buffer){
    int i, j, np  = nc / NR, _nr = nc % NR;

    for (j=0; j<np; ++j) {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    if (_nr>0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (j=_nr; j<NR; ++j) {
                buffer[j] = UBIT;
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}


//
//  Micro kernel for multiplying panels from A and B.
//
static void
dgemm_micro_kernel(int m, int n, int kc, uint32_t *A, uint32_t *B, int beta, float *C, int incRowC, int incColC)
{
    int AB[MR*NR];
    int i, j, l;

//
//  Compute AB = A*B
//
    //#pragma omp parallel for
    for (l=0; l<MR*NR; ++l) {
        AB[l] = 0;
    }

    for (l=0; l<kc; ++l) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                AB[i+j*MR] -= popcnt32(MASK(A[i]^B[j]))<<1;
            }
        }
        A += MR;
        B += NR;
    }

//
//  Update C <- beta*C
//
    if (!beta) {
        //#pragma omp for collapse(2)
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                C[i*incRowC+j*incColC] += 0;
            }
        }
    }
    //#pragma omp for collapse(2)
    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            C[i*incRowC+j*incColC] += AB[i+j*MR];
        }
    }
}

//
//  Compute X *= alpha
//
static void
dgescal(int     m,
        int     n,
        int  beta,
        float  *X,
        int     incRowX,
        int     incColX)
{
    int i, j;

    if (!beta) {
        //#pragma omp parallel for schedule(dynamic,1) collapse(2)
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0;
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   int     beta,
                   float  *C,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;
    #pragma omp parallel shared(C) private(i,j,nr,mr)
    {
        #pragma omp for schedule(dynamic)
        for (j=0; j<np; ++j) {
            nr    = (j!=np-1 || _nr==0) ? NR : _nr;
            for (i=0; i<mp; ++i) {
                mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

                if (mr==MR && nr==NR) {

                    dgemm_micro_kernel(mr, nr, kc, &_A[i*kc*MR], &_B[j*kc*NR], beta, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);

                } else {

                    dgescal(mr, nr, beta, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);

                    dgemm_micro_kernel(mr, nr, kc, &_A[i*kc*MR], &_B[j*kc*NR], 0, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);

                }
            }
        }
    }
}

//
//  Compute C <- beta*C + A*B, beta = 0 or 1
//
void
dgemm_nn(int            m,
         int            n,
         int            kk,
         uint32_t      *A,
         int            incRowA,
         int            incColA,
         uint32_t      *B,
         int            incRowB,
         int            incColB,
         float         *C,
         int            incRowC,
         int            incColC,
         int            beta,
         int            alpha,
         float         *alphas)
{
    int i, j, l, k = 1+(kk-1)/ENCODE_BIT;
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;

    int _beta;

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            _beta = (l==0) ? beta : 1;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                       _A);

                dgemm_macro_kernel(mc, nc, kc, _beta,
                                   &C[i*MC*incRowC+j*NC*incColC],
                                   incRowC, incColC);
            }
        }
    }

    //#pragma omp parallel for schedule(dynamic,1) collapse(2)
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            C[i*n+j]+=kk;
        }
    }
    if(alpha){
        for(i=0; i<m; i++){
            for(j=0; j<n; j++){
                C[i*n+j]*=alphas[i];
            }
        }
    }
}

#endif /* MATMUL_H */

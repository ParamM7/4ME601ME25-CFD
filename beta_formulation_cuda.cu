//to execute: nvcc -arch=sm_89 -std=c++17 -Xcompiler "/O2 /EHsc /Zc:preprocessor" -o heat2d_cuda.exe beta_formulation_cuda.cu
// .\heat2d_cuda.exe [beta] [ni] [nj]

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <chrono>
#include <new>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

//__CUDACC__ must be tested FIRST. On Windows nvcc emulates MSVC's predefined
//macros while compiling device code, so a _MSC_VER-first test would expand
//RESTRICT to MSVC's __restrict inside every kernel, which cicc rejects.
#if defined(__CUDACC__)
    #define RESTRICT __restrict__
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT
#endif

#if defined(_WIN32)
    #include <direct.h>
    #define MAKE_DIR(p) _mkdir(p)
#else
    #include <sys/stat.h>
    #define MAKE_DIR(p) mkdir(p, 0755)
#endif

using namespace std;

//Kernel launches are asynchronous and fail silently. Without this an
//out-of-bounds write shows up as a wrong number three functions later.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d -> %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(e_));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

//A device_vector is a HOST-side handle: the container lives on the host, the
//buffer lives in VRAM. It cannot appear inside a kernel, so every launch site
//passes RAW(V) instead of V.
#define RAW(V) thrust::raw_pointer_cast((V).data())

//Lets std::vector own page-locked memory. Pageable memory forces the driver to
//stage every cudaMemcpy through an internal bounce buffer, roughly halving
//effective PCIe bandwidth.
template <class Tp>
struct PINNED_ALLOCATOR
{
    typedef Tp value_type;
    PINNED_ALLOCATOR() noexcept {}
    template <class U> PINNED_ALLOCATOR(const PINNED_ALLOCATOR<U>&) noexcept {}

    Tp* allocate(std::size_t n)
    {
        void* p = NULL;
        if (cudaMallocHost(&p, n*sizeof(Tp)) != cudaSuccess) throw std::bad_alloc();
        return static_cast<Tp*>(p);
    }
    void deallocate(Tp* p, std::size_t) noexcept { cudaFreeHost(p); }
};
template <class A, class B>
bool operator==(const PINNED_ALLOCATOR<A>&, const PINNED_ALLOCATOR<B>&) { return true; }
template <class A, class B>
bool operator!=(const PINNED_ALLOCATOR<A>&, const PINNED_ALLOCATOR<B>&) { return false; }

using Field       = thrust::device_vector<double>;
using HostField   = std::vector<double>;
using PinnedField = std::vector<double, PINNED_ALLOCATOR<double> >;

int NJ = 100;
int NI = 100;

inline size_t ID(int j, int i) { return (size_t)j*NI + i;}
inline size_t IDX(int c, int j, int i) { return ((size_t)c*NJ + j)*NI + i;}

//AW..AP1 come out spatially uniform for every BETA (see CALC_COEFF_TRANSIENT --
//no i or j appears on any right-hand side). Holding them as ten scalars in
//constant memory instead of ten NI*NJ arrays cuts the stencil from 15 global
//loads per node to 5 and drops the footprint from 136 to 56 bytes/node.
//Set to 0 the moment the coefficients become space-dependent.
#define UNIFORM_COEFF 1

//Captures the CG inner loop as a CUDA graph. Legal because none of the buffers
//the five CG kernels touch are ever swapped -- only T and T_old_time swap, and
//no CG kernel reads either one. Worth most under WDDM, where launch overhead
//runs 5-10 us instead of the ~3 us seen on Linux.
#define USE_CG_GRAPH 1

void SET_GEOMETRY();

void APPLYIC_CG();
void APPLYBC_TEMP_CG();
void CALC_RESIDUAL_CG();
void SOLVER_CG();

void APPLYIC_TRANSIENT();
void CALC_COEFF_TRANSIENT();
void BUILD_RHS_TRANSIENT();
void SOLVER_TRANSIENT();
void WRITE_FILE_TRANSIENT();
void CHECK_STABILITY();
void SET_DELTAT();
void CALC_FOURIER();
void APPLYBC_TEMP_TRANSIENT();
double UPDATE_TRANSIENT();
void WRITE_FILE_TRANSIENT_VTK();

double CALC_NORM_FIELD(const Field& ARR);

double FO_LIMIT();
void   MAKE_OUTPUT_DIR();
void   PUSH_PARAMS();
void   ALLOCATE_FIELDS();
void   FREE_FIELDS();
void   FETCH_FIELD();
void   BUILD_CG_GRAPH();
void   DESTROY_CG_GRAPH();

double BETA = 0.5;

#define OUTPUT_DIR "results"

#define BETA_ZERO_TOL 1.0e-14

bool VERBOSE = false;

HostField   XCELL;      //host only, geometry is used for output and nothing else
PinnedField T_HOST;     //host staging buffer for the file writes

Field SP;
Field T, T_old_time;
Field AW, AE, AS, AN, AP;
Field AW1, AE1, AS1, AN1, AP1;
Field TCG, RES, PDIR, AP_CG;
Field PARTIAL;          //per-block partials of r.r, drives RS_OLD and RS_NEW
Field PARTIAL_N;        //per-block partials of the norm, drives RRMS_CG

double DELX, DELY;

int NCELLI;
int NCELLJ;

double LX,LY;
double CCSS; //convergence criteria
double RRMS, RMSRESIDUE;
long TOTAL_CG_ITER = 0;

//launch configuration, fixed once in ALLOCATE_FIELDS and never touched again
dim3 BLK_INT, GRD_INT, GRD_ALL;
int  NBLK_INT = 0;

#define NTHREAD_RED 256     //1-D block used by the second-stage reductions
#define CHECK_EVERY 50      //CG iterations per graph replay / host read of DONE_CG
#define CG_CHUNK    CHECK_EVERY

//NI/NJ are runtime, so the kernels cannot use the host ID(). Constant memory is
//broadcast to a whole warp at register speed, unlike a global load.
struct PARAMS
{
    int    NI, NJ, NCELLI, NCELLJ;
    double DELX, DELY, deltaT, invdt;
};
__constant__ PARAMS PAR;
PARAMS H_PAR;

__device__ __forceinline__ size_t ID_D(int j, int i) { return (size_t)j*PAR.NI + i; }

struct UNIFORM_COEFFS
{
    double AW, AE, AS, AN, AP;          //acts on level n+1, the operator A
    double AW1, AE1, AS1, AN1, AP1;     //acts on level n, the RHS assembly
};
__constant__ UNIFORM_COEFFS UCO;
UNIFORM_COEFFS H_UCO;

#if UNIFORM_COEFF
    #define CF_W(p)  UCO.AW
    #define CF_E(p)  UCO.AE
    #define CF_S(p)  UCO.AS
    #define CF_N(p)  UCO.AN
    #define CF_P(p)  UCO.AP
    #define CF_W1(p) UCO.AW1
    #define CF_E1(p) UCO.AE1
    #define CF_S1(p) UCO.AS1
    #define CF_N1(p) UCO.AN1
    #define CF_P1(p) UCO.AP1
#else
    #define CF_W(p)  aw[p]
    #define CF_E(p)  ae[p]
    #define CF_S(p)  as[p]
    #define CF_N(p)  an[p]
    #define CF_P(p)  ap[p]
    #define CF_W1(p) aw1[p]
    #define CF_E1(p) ae1[p]
    #define CF_S1(p) as1[p]
    #define CF_N1(p) an1[p]
    #define CF_P1(p) ap1[p]
#endif

//Every CG scalar lives on the DEVICE. Copying ALPHA_CG back each iteration
//would be an implicit device sync -- at 315086 iterations (400^2, BETA=0.5)
//that is seconds of stalling before any useful work happens.
struct CG_SCALARS
{
    double RS_OLD;
    double RS_NEW;
    double PAP_CG;
    double ALPHA_CG;
    double BETA_CG;
    double RRMS_CG;
    double DTDT;
    double NORM;                //result slot for the standalone CALC_NORM_FIELD
    long long TOTAL_CG_ITER;
    int    DONE_CG;             //0 running, 1 converged, 2 breakdown
};
__device__ CG_SCALARS CG;
CG_SCALARS H_CG;

//host mirrors, read at the check interval for printing and the MAXITER warning
double ALPHA_CG, BETA_CG, RS_OLD, RS_NEW, PAP_CG, RRMS_CG;
int    ITER_CG;

#define MAXITER_CG 50000

double simTime;

int TIMESTEP;

#define T_WEST   10.0
#define T_SOUTH  20.0
#define T_EAST   10.0
#define T_NORTH  70.0
#define T_INIT   10.0

#define A_WEST    0.0
#define B_WEST    1.0
#define C_WEST    T_WEST

#define A_SOUTH  -10.0
#define B_SOUTH   1.0
#define C_SOUTH   T_SOUTH

#define A_EAST    0.0
#define B_EAST    1.0
#define C_EAST    T_EAST

#define A_NORTH   0.0
#define B_NORTH   1.0
#define C_NORTH   T_NORTH

#define kT 1.0 //thermal conductivity
#define rho 1.0 //density
#define Cp 1.0 //specific heat

double deltaT;

double alpha = kT/(rho*Cp);

double Fo_x, Fo_y; //Fourier number

#define TMAX        1.0     //stop time; the diffusion timescale L^2/alpha is 1 here
#define MAXSTEP     200000  //hard cap on physical timesteps
double WRITE_INTERVAL = 1e-2; //time interval between writes
double nextWriteTime = WRITE_INTERVAL;   //dump a field file every N timesteps
#define DT_SAFETY   0.8     //BETA < 0.5: fraction of the conditional stability limit
#define DT_ACCURACY 1.0e-3  //BETA >= 0.5: dt chosen for accuracy
#define STEADY_TOL  1.0e-9
double io_s = 0.0;

//The norm splits in two on the GPU: NORM_TERM is the per-node contribution,
//evaluated inside the fused sweeps that are already touching RES, and
//CALC_NORM_Lx is the finalizer applied to the reduced sum -- the tail of the
//CPU CALC_NORM_Lx(). Switching NORM_TYPE therefore costs no extra sweep. With
//NORM_L2 the norm sum IS the r.r sum CG already needs and the second
//accumulator compiles away entirely; with NORM_L1 one extra block reduction
//rides along, still with no extra pass over global memory.
//
//RS_OLD and RS_NEW stay L2 in both modes: CG needs r.r for BETA_CG, and an L1
//quantity there would break A-conjugacy. Only the convergence TEST follows
//NORM_TYPE, same as the CPU version.

#define NORM_L1 1
#define NORM_L2 2

#define NORM_TYPE NORM_L2

#if NORM_TYPE == NORM_L1
    #define CALC_NORM CALC_NORM_L1
#else
    #define CALC_NORM CALC_NORM_L2
#endif

__device__ __forceinline__ double CALC_NORM_L2(double SUMSQ, double invN)
{
    return sqrt(SUMSQ*invN);
}

__device__ __forceinline__ double CALC_NORM_L1(double SUMABS, double invN)
{
    return SUMABS*invN;
}

__device__ __forceinline__ double NORM_TERM(double v)
{
#if NORM_TYPE == NORM_L1
    return fabs(v);
#else
    return v*v;
#endif
}

//__shfl_down_sync moves a register straight between lanes of a warp: no shared
//memory, no __syncthreads, 5 instructions for 32 values.
__inline__ __device__ double WARP_REDUCE_SUM(double v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

//Serves both the 2-D interior blocks and the 1-D reduction blocks.
__inline__ __device__ double BLOCK_REDUCE_SUM(double v)
{
    __shared__ double SH[32];

    const int TID  = threadIdx.y*blockDim.x + threadIdx.x;
    const int NT   = blockDim.x*blockDim.y;
    const int LANE = TID & 31;
    const int WID  = TID >> 5;

    v = WARP_REDUCE_SUM(v);
    if (LANE == 0) SH[WID] = v;
    __syncthreads();

    v = (TID < ((NT + 31) >> 5)) ? SH[TID] : 0.0;
    if (WID == 0) v = WARP_REDUCE_SUM(v);
    return v;
}

//Two sums through ONE barrier. Calling BLOCK_REDUCE_SUM twice back to back
//would race on its shared array -- silently wrong numbers, not a crash.
__inline__ __device__ void BLOCK_REDUCE_SUM2(double& a, double& b)
{
    __shared__ double SHA[32];
    __shared__ double SHB[32];

    const int TID  = threadIdx.y*blockDim.x + threadIdx.x;
    const int NT   = blockDim.x*blockDim.y;
    const int LANE = TID & 31;
    const int WID  = TID >> 5;

    a = WARP_REDUCE_SUM(a);
    b = WARP_REDUCE_SUM(b);
    if (LANE == 0) { SHA[WID] = a; SHB[WID] = b; }
    __syncthreads();

    const int NW = (NT + 31) >> 5;
    a = (TID < NW) ? SHA[TID] : 0.0;
    b = (TID < NW) ? SHB[TID] : 0.0;
    if (WID == 0) { a = WARP_REDUCE_SUM(a); b = WARP_REDUCE_SUM(b); }
}

__device__ __forceinline__ int BLOCK_ID_2D() { return blockIdx.y*gridDim.x + blockIdx.x; }
__device__ __forceinline__ int THREAD_ID_2D(){ return threadIdx.y*blockDim.x + threadIdx.x; }

//Second stage: one block folds the NB block-partials into a scalar. A
//fixed-size tree, so the result is bit-reproducible run to run. atomicAdd on
//doubles would not be, and a nondeterministic dot product means a
//nondeterministic CG iteration count.
__inline__ __device__ double REDUCE_PARTIAL(const double* RESTRICT part, int NB)
{
    double s = 0.0;
    for (int b = threadIdx.x; b < NB; b += blockDim.x) s += part[b];
    return BLOCK_REDUCE_SUM(s);
}

__inline__ __device__ void REDUCE_PARTIAL2(const double* RESTRICT pa,
                                           const double* RESTRICT pb,
                                           int NB, double& ra, double& rb)
{
    double a = 0.0, b = 0.0;
    for (int i = threadIdx.x; i < NB; i += blockDim.x) { a += pa[i]; b += pb[i]; }
    BLOCK_REDUCE_SUM2(a, b);
    ra = a; rb = b;
}

__global__ void CALC_NORM_K(const double* RESTRICT arr, double* RESTRICT part)
{
    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;

    double s = 0.0;
    if (i < PAR.NCELLI && j < PAR.NCELLJ) s = NORM_TERM(arr[ID_D(j,i)]);

    s = BLOCK_REDUCE_SUM(s);
    if (THREAD_ID_2D() == 0) part[BLOCK_ID_2D()] = s;
}

__global__ void FINALIZE_NORM_K(const double* RESTRICT part, int NB, double invN)
{
    const double s = REDUCE_PARTIAL(part, NB);
    if (threadIdx.x == 0) CG.NORM = CALC_NORM(s, invN);
}

__global__ void APPLYIC_TRANSIENT_K(double* RESTRICT t, double* RESTRICT told)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= PAR.NI || j >= PAR.NJ) return;

    const size_t p = ID_D(j,i);
    t[p]    = T_INIT;
    told[p] = T_INIT;
}

//Generalized boundary condition c+a(dT/dn)=bT.
//
//The four edges are not independent at the corners. The west loop runs
//j=0..NJ-1 so it writes T[ID(0,0)] from T[ID(0,1)], a south-boundary node the
//south loop overwrites two lines later; the east loop, running third, reads
//T[ID(0,NCELLI-1)], which the south loop HAS already updated. Launch all four
//edges concurrently and those corners become a data race. One block with
//__syncthreads() between the stages reproduces the CPU ordering exactly.
//Corner values never enter the solution but they do get written to the files.
__global__ void APPLYBC_TEMP_K(double* RESTRICT F)
{
    const int    ni     = PAR.NI;
    const int    nj     = PAR.NJ;
    const int    ncelli = PAR.NCELLI;
    const int    ncellj = PAR.NCELLJ;
    const double delx   = PAR.DELX;
    const double dely   = PAR.DELY;

    //west
    for (int j = threadIdx.x; j < nj; j += blockDim.x)
        F[(size_t)j*ni] = (A_WEST*F[(size_t)j*ni + 1] + C_WEST*delx)
                        / (A_WEST + B_WEST*delx);
    __syncthreads();

    //south
    for (int i = 1 + threadIdx.x; i < ncelli; i += blockDim.x)
        F[(size_t)i] = (A_SOUTH*F[(size_t)ni + i] + C_SOUTH*dely)
                     / (A_SOUTH + B_SOUTH*dely);
    __syncthreads();

    //east
    for (int j = threadIdx.x; j < nj; j += blockDim.x)
        F[(size_t)j*ni + ncelli] = (A_EAST*F[(size_t)j*ni + ncelli-1] + C_EAST*delx)
                                 / (A_EAST + B_EAST*delx);
    __syncthreads();

    //north
    for (int i = 1 + threadIdx.x; i < ncelli; i += blockDim.x)
        F[(size_t)ncellj*ni + i] = (A_NORTH*F[(size_t)(ncellj-1)*ni + i] + C_NORTH*dely)
                                 / (A_NORTH + B_NORTH*dely);
}

__global__ void CALC_COEFF_TRANSIENT_K(double* RESTRICT aw, double* RESTRICT ae,
                                       double* RESTRICT as, double* RESTRICT an,
                                       double* RESTRICT ap,
                                       double* RESTRICT aw1, double* RESTRICT ae1,
                                       double* RESTRICT as1, double* RESTRICT an1,
                                       double* RESTRICT ap1,
                                       double BETA, double Fo_x, double Fo_y)
{
    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= PAR.NCELLI || j >= PAR.NCELLJ) return;

    const size_t p = ID_D(j,i);

    aw[p] = BETA*Fo_x;
    ae[p] = BETA*Fo_x;
    an[p] = BETA*Fo_y;
    as[p] = BETA*Fo_y;

    ap[p] = 1.0 + 2.0*BETA*(Fo_x + Fo_y);

    aw1[p] = (1.0-BETA)*Fo_x;
    ae1[p] = (1.0-BETA)*Fo_x;
    an1[p] = (1.0-BETA)*Fo_y;
    as1[p] = (1.0-BETA)*Fo_y;

    ap1[p] = 1.0 - 2.0*(1.0-BETA)*(Fo_x + Fo_y);
}

__global__ void BUILD_RHS_TRANSIENT_K(const double* RESTRICT ap1,
                                      const double* RESTRICT aw1,
                                      const double* RESTRICT ae1,
                                      const double* RESTRICT as1,
                                      const double* RESTRICT an1,
                                      const double* RESTRICT told,
                                      double* RESTRICT sp)
{
    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= PAR.NCELLI || j >= PAR.NCELLJ) return;

    const int    ni = PAR.NI;
    const size_t p  = ID_D(j,i);

    sp[p] = CF_P1(p)*told[p]
          + CF_W1(p)*told[p-1]  + CF_E1(p)*told[p+1]
          + CF_S1(p)*told[p-ni] + CF_N1(p)*told[p+ni];
}

__global__ void COPY_INTERIOR_K(double* RESTRICT dst, const double* RESTRICT src)
{
    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= PAR.NCELLI || j >= PAR.NCELLJ) return;

    const size_t p = ID_D(j,i);
    dst[p] = src[p];
}

__global__ void APPLYIC_CG_K(double* RESTRICT tcg, double* RESTRICT res,
                             double* RESTRICT pdir, double* RESTRICT apcg,
                             const double* RESTRICT told)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= PAR.NI || j >= PAR.NJ) return;

    const size_t p = ID_D(j,i);

    tcg[p]  = told[p];      //warm start
    res[p]  = 0.0;
    pdir[p] = 0.0;
    apcg[p] = 0.0;
}

//r = b-Ax, p0 = r0, and the partials for r.r and the norm, all in one sweep.
__global__ void CALC_RESIDUAL_CG_K(const double* RESTRICT ap, const double* RESTRICT aw,
                                   const double* RESTRICT ae, const double* RESTRICT as,
                                   const double* RESTRICT an,
                                   const double* RESTRICT tcg, const double* RESTRICT sp,
                                   double* RESTRICT res, double* RESTRICT pdir,
                                   double* RESTRICT part, double* RESTRICT partn)
{
    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;

    //Guard the computation, never return early. The block reductions below
    //contain __syncthreads(), and out-of-range threads that have returned
    //would hang the in-range ones. At 100^2 with a 32x8 block, 30 threads per
    //row-block are out of range on every single launch.
    double s = 0.0;
#if NORM_TYPE == NORM_L1
    double sn = 0.0;
#endif

    if (i < PAR.NCELLI && j < PAR.NCELLJ)
    {
        const int    ni = PAR.NI;
        const size_t p  = ID_D(j,i);

        const double Ax = CF_P(p)*tcg[p]
                        - CF_W(p)*tcg[p-1]  - CF_E(p)*tcg[p+1]
                        - CF_S(p)*tcg[p-ni] - CF_N(p)*tcg[p+ni];

        const double r = sp[p] - Ax;
        res[p]  = r;
        pdir[p] = r;
        s = r*r;
#if NORM_TYPE == NORM_L1
        sn = NORM_TERM(r);
#endif
    }

#if NORM_TYPE == NORM_L1
    BLOCK_REDUCE_SUM2(s, sn);
    if (THREAD_ID_2D() == 0) { part[BLOCK_ID_2D()] = s; partn[BLOCK_ID_2D()] = sn; }
#else
    s = BLOCK_REDUCE_SUM(s);        //the r.r sum IS the L2 norm sum
    if (THREAD_ID_2D() == 0) part[BLOCK_ID_2D()] = s;
#endif
}

__global__ void CALC_RS_OLD_K(const double* RESTRICT part, const double* RESTRICT partn,
                              int NB, double invN, double CCSS)
{
#if NORM_TYPE == NORM_L1
    double s = 0.0, sn = 0.0;
    REDUCE_PARTIAL2(part, partn, NB, s, sn);
#else
    const double s  = REDUCE_PARTIAL(part, NB);
    const double sn = s;
#endif

    if (threadIdx.x == 0)
    {
        //With the warm start TCG = T^n the initial residual can already sit
        //below tolerance. Entering the loop would then divide RS_OLD by a
        //vanishing PAP_CG and push inf/NaN straight into TCG.
        CG.RS_OLD  = s;
        CG.RRMS_CG = CALC_NORM(sn, invN);
        CG.DONE_CG = (CG.RRMS_CG < CCSS) ? 1 : 0;
    }
}

//Ap = A*p with p.Ap accumulated in the same sweep. Matrix-free: A is never
//stored, only applied as a 5-point stencil. PDIR is zero on the boundary
//(APPLYIC_CG zeroes the full field and only interior nodes are ever written),
//so those neighbour terms drop out cleanly, which is exactly the homogeneous
//Dirichlet operator CG needs.
__global__ void CALC_AP_CG_K(const double* RESTRICT ap, const double* RESTRICT aw,
                             const double* RESTRICT ae, const double* RESTRICT as,
                             const double* RESTRICT an,
                             const double* RESTRICT pd, double* RESTRICT apcg,
                             double* RESTRICT part)
{
    if (CG.DONE_CG) return;     //predicated no-op once converged

    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;

    double s = 0.0;
    if (i < PAR.NCELLI && j < PAR.NCELLJ)
    {
        const int    ni = PAR.NI;
        const size_t p  = ID_D(j,i);

        const double v = CF_P(p)*pd[p]
                       - CF_W(p)*pd[p-1]  - CF_E(p)*pd[p+1]
                       - CF_S(p)*pd[p-ni] - CF_N(p)*pd[p+ni];

        apcg[p] = v;
        s = pd[p]*v;            //v is still in a register, so the dot is free
    }
    s = BLOCK_REDUCE_SUM(s);
    if (THREAD_ID_2D() == 0) part[BLOCK_ID_2D()] = s;
}

__global__ void CALC_ALPHA_CG_K(const double* RESTRICT part, int NB)
{
    if (CG.DONE_CG) return;

    const double pap = REDUCE_PARTIAL(part, NB);

    if (threadIdx.x == 0)
    {
        CG.PAP_CG = pap;

        //For an SPD matrix p.Ap > 0 strictly. Zero means either convergence to
        //machine precision or a broken (non-symmetric) operator -- stop either way.
        if (fabs(pap) < 1.0e-30) { CG.DONE_CG = 2; return; }

        CG.ALPHA_CG = CG.RS_OLD/pap;
        CG.TOTAL_CG_ITER++;
    }
}

//Update solution x and residual r simultaneously. The recurrence
//r <- r - alpha*Ap avoids recomputing b - Ax, saving a full stencil sweep.
__global__ void UPDATE_XR_CG_K(double* RESTRICT tcg, double* RESTRICT res,
                               const double* RESTRICT pd, const double* RESTRICT apcg,
                               double* RESTRICT part, double* RESTRICT partn)
{
    if (CG.DONE_CG) return;

    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;

    const double ALPHA_CG = CG.ALPHA_CG;

    double s = 0.0;
#if NORM_TYPE == NORM_L1
    double sn = 0.0;
#endif

    if (i < PAR.NCELLI && j < PAR.NCELLJ)
    {
        const size_t p = ID_D(j,i);

        tcg[p] = tcg[p] + ALPHA_CG*pd[p];

        const double r = res[p] - ALPHA_CG*apcg[p];
        res[p] = r;
        s = r*r;
#if NORM_TYPE == NORM_L1
        sn = NORM_TERM(r);
#endif
    }

#if NORM_TYPE == NORM_L1
    BLOCK_REDUCE_SUM2(s, sn);
    if (THREAD_ID_2D() == 0) { part[BLOCK_ID_2D()] = s; partn[BLOCK_ID_2D()] = sn; }
#else
    s = BLOCK_REDUCE_SUM(s);
    if (THREAD_ID_2D() == 0) part[BLOCK_ID_2D()] = s;
#endif
}

__global__ void CALC_BETA_CG_K(const double* RESTRICT part, const double* RESTRICT partn,
                               int NB, double invN, double CCSS)
{
    if (CG.DONE_CG) return;

#if NORM_TYPE == NORM_L1
    double rs_new = 0.0, sn = 0.0;
    REDUCE_PARTIAL2(part, partn, NB, rs_new, sn);
#else
    const double rs_new = REDUCE_PARTIAL(part, NB);
    const double sn     = rs_new;
#endif

    if (threadIdx.x == 0)
    {
        CG.RS_NEW = rs_new;

        //Under NORM_L2 this is the sum CALC_NORM_L2 would recompute from RES,
        //same terms -- which is why the CPU's separate norm sweep is gone.
        CG.RRMS_CG = CALC_NORM(sn, invN);

        //Break BEFORE BETA_CG, PDIR and RS_OLD are touched, matching the CPU.
        if (CG.RRMS_CG < CCSS) { CG.DONE_CG = 1; return; }

        //beta is what keeps successive directions A-conjugate; drop it and this
        //degenerates into steepest descent.
        CG.BETA_CG = rs_new/CG.RS_OLD;
        CG.RS_OLD  = rs_new;
    }
}

__global__ void UPDATE_PDIR_K(double* RESTRICT pd, const double* RESTRICT res)
{
    if (CG.DONE_CG) return;

    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= PAR.NCELLI || j >= PAR.NCELLJ) return;

    const size_t p = ID_D(j,i);
    pd[p] = res[p] + CG.BETA_CG*pd[p];
}

__global__ void UPDATE_TRANSIENT_K(const double* RESTRICT t, const double* RESTRICT told,
                                   double* RESTRICT part)
{
    const int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    const int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;

    double s = 0.0;
    if (i < PAR.NCELLI && j < PAR.NCELLJ)
    {
        const size_t p = ID_D(j,i);
        const double d = (t[p] - told[p])*PAR.invdt;
        s = d*d;
    }
    s = BLOCK_REDUCE_SUM(s);
    if (THREAD_ID_2D() == 0) part[BLOCK_ID_2D()] = s;
}

__global__ void CALC_DTDT_K(const double* RESTRICT part, int NB, double invN)
{
    const double s = REDUCE_PARTIAL(part, NB);

    //Deliberately L2 regardless of NORM_TYPE: the CPU UPDATE_TRANSIENT
    //hardcodes sqrt(sumsq*invN) and never routed through CALC_NORM.
    if (threadIdx.x == 0) CG.DTDT = CALC_NORM_L2(s, invN);
}

cudaStream_t    STREAM_CG = NULL;
cudaGraph_t     G_CG      = NULL;
cudaGraphExec_t E_CG      = NULL;

static void LAUNCH_CG_ITER(cudaStream_t S)
{
    const double invN = 1.0/((double)(NI-2)*(double)(NJ-2));

    CALC_AP_CG_K   <<<GRD_INT, BLK_INT, 0, S>>>(RAW(AP), RAW(AW), RAW(AE), RAW(AS),
                                                RAW(AN), RAW(PDIR), RAW(AP_CG),
                                                RAW(PARTIAL));

    CALC_ALPHA_CG_K<<<1, NTHREAD_RED, 0, S>>>(RAW(PARTIAL), NBLK_INT);

    UPDATE_XR_CG_K <<<GRD_INT, BLK_INT, 0, S>>>(RAW(TCG), RAW(RES), RAW(PDIR),
                                                RAW(AP_CG), RAW(PARTIAL), RAW(PARTIAL_N));

    CALC_BETA_CG_K <<<1, NTHREAD_RED, 0, S>>>(RAW(PARTIAL), RAW(PARTIAL_N),
                                              NBLK_INT, invN, CCSS);

    UPDATE_PDIR_K  <<<GRD_INT, BLK_INT, 0, S>>>(RAW(PDIR), RAW(RES));
}

//Captured once, after the device pointers are final. None of the buffers above
//are ever swapped, so the graph stays valid for the whole run.
void BUILD_CG_GRAPH()
{
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamCreate(&STREAM_CG));

#if USE_CG_GRAPH
    CUDA_CHECK(cudaStreamBeginCapture(STREAM_CG, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < CG_CHUNK; k++) LAUNCH_CG_ITER(STREAM_CG);
    CUDA_CHECK(cudaStreamEndCapture(STREAM_CG, &G_CG));
    CUDA_CHECK(cudaGraphInstantiate(&E_CG, G_CG, NULL, NULL, 0));

    cout << "GRAPH   : CG chunk of " << CG_CHUNK << " iterations captured" << endl;
#endif
}

void DESTROY_CG_GRAPH()
{
#if USE_CG_GRAPH
    if (E_CG) { cudaGraphExecDestroy(E_CG); E_CG = NULL; }
    if (G_CG) { cudaGraphDestroy(G_CG);     G_CG = NULL; }
#endif
    if (STREAM_CG) { cudaStreamDestroy(STREAM_CG); STREAM_CG = NULL; }
}

int main(int argc, char* argv[]){
    const auto t_wall_start = std::chrono::steady_clock::now();
    CCSS = 1e-13;
    LX = 1.0;
    LY = 1.0;

    NCELLI = NI-1;
    NCELLJ = NJ-1;

    if(argc > 1)
    {
        BETA = atof(argv[1]);

        if(BETA< 0.0 || BETA > 1.0)
        {
            cout << "BETA must be in [0,1] for stability." << endl;
            return -1;
        }
    }

    if(argc > 3)
    {
        NI = atoi(argv[2]);
        NJ = atoi(argv[3]);

        if(NI < 3 || NJ < 3)
        {
            cout<<"NI and NJ must both be >= 3"<<endl;
            return -1;
        }
    }

    NCELLI = NI-1;
    NCELLJ = NJ-1;

    int DEV = 0;
    CUDA_CHECK(cudaSetDevice(DEV));
    cudaDeviceProp PROP;
    CUDA_CHECK(cudaGetDeviceProperties(&PROP, DEV));

    cout<< "DEVICE  : " << PROP.name << "  sm_" << PROP.major << PROP.minor
        << "   SMs = " << PROP.multiProcessorCount << endl;

    cout<< "GRID: NI = " << NI << "   NJ = " << NJ << endl;
    cout<< "    ("<< (size_t)NI*NJ << "nodes)" <<endl;
    cout<< "NORM    : " << (NORM_TYPE == NORM_L1 ? "L1" : "L2")
        << "   coefficients: " << (UNIFORM_COEFF ? "uniform (constant memory)"
                                                 : "array") << endl;

    ALLOCATE_FIELDS();

    MAKE_OUTPUT_DIR();

    SET_GEOMETRY();

    SET_DELTAT();

    CALC_FOURIER();
    CHECK_STABILITY();

    APPLYIC_TRANSIENT();
    APPLYBC_TEMP_TRANSIENT();

    CALC_COEFF_TRANSIENT();

    if(BETA >= BETA_ZERO_TOL) BUILD_CG_GRAPH();

    TIMESTEP = 0;
    simTime = 0.0;

    FETCH_FIELD();
    WRITE_FILE_TRANSIENT();

    while(simTime < TMAX && TIMESTEP <MAXSTEP)
    {
        //device_vector::swap exchanges the internal device pointers, so no
        //bytes move. Every element of T is rewritten below before it is read:
        //SOLVER_TRANSIENT writes the interior, APPLYBC_TEMP_TRANSIENT the edges.
        T.swap(T_old_time);

        SOLVER_TRANSIENT();

        APPLYBC_TEMP_TRANSIENT();

        double dTdt = UPDATE_TRANSIENT();

        simTime = simTime + deltaT;
        TIMESTEP = TIMESTEP + 1;

        if(TIMESTEP % 100 == 0)
        {
            cout << "STEP = " << setw(6) << TIMESTEP
                 << "   t = " << fixed << setprecision(6) << simTime
                 << "   ||dT/dt|| = " << scientific << setprecision(4) << dTdt
                 << endl;
        }

        if(simTime + 0.5*deltaT >= nextWriteTime)
        {
            const auto t_io0 = std::chrono::steady_clock::now();

            FETCH_FIELD();
            WRITE_FILE_TRANSIENT();
            WRITE_FILE_TRANSIENT_VTK();

            io_s += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_io0).count();

            nextWriteTime += WRITE_INTERVAL;
        }
        /*
        if(dTdt < STEADY_TOL)
        {
            cout << "Steady state reached at t = " << simTime
                 << " after " << TIMESTEP << " timesteps." << endl;
            break;
        }*/
    }

    {
        const auto t_io0 = std::chrono::steady_clock::now();

        FETCH_FIELD();
        WRITE_FILE_TRANSIENT();
        WRITE_FILE_TRANSIENT_VTK();

        io_s += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_io0).count();
    }

    if(BETA >= BETA_ZERO_TOL)
    {
        CUDA_CHECK(cudaMemcpyFromSymbol(&H_CG, CG, sizeof(CG_SCALARS)));
        TOTAL_CG_ITER = (long)H_CG.TOTAL_CG_ITER;

        cout << "Total CG iterations = " << TOTAL_CG_ITER
            << "   (average " << (double)TOTAL_CG_ITER/(double)(TIMESTEP>0?TIMESTEP:1)
            << " per timestep)" << endl;
    }

    const auto t_wall_end = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t_wall_end - t_wall_start).count();

    cout << "WALLTIME: total = " << fixed << setprecision(3) << wall_s << " s"
         << "   I/O = " << io_s << " s"
         << "   compute = " << wall_s - io_s << " s" << endl;
    cout << "          (" << TIMESTEP << " steps, "
         << scientific << setprecision(3)
         << (double)TIMESTEP*(double)(NI-2)*(double)(NJ-2)/(wall_s - io_s)
         << " node-updates/s compute-only)" << endl;

    FREE_FIELDS();

    return (0);

}

void PUSH_PARAMS()
{
    H_PAR.NI     = NI;
    H_PAR.NJ     = NJ;
    H_PAR.NCELLI = NCELLI;
    H_PAR.NCELLJ = NCELLJ;
    H_PAR.DELX   = DELX;
    H_PAR.DELY   = DELY;
    H_PAR.deltaT = deltaT;
    H_PAR.invdt  = (deltaT != 0.0) ? 1.0/deltaT : 0.0;

    CUDA_CHECK(cudaMemcpyToSymbol(PAR, &H_PAR, sizeof(PARAMS)));
}

void ALLOCATE_FIELDS()
{
    const size_t N = (size_t)NI*NJ;

    XCELL.assign(2*N, 0.0);
    T_HOST.assign(N, 0.0);

    SP.assign(N, 0.0);

    T.assign(N, 0.0);
    T_old_time.assign(N, 0.0);

#if UNIFORM_COEFF == 0
    AW.assign(N, 0.0);
    AE.assign(N, 0.0);
    AN.assign(N, 0.0);
    AS.assign(N, 0.0);
    AP.assign(N, 0.0);

    AW1.assign(N, 0.0);
    AE1.assign(N, 0.0);
    AN1.assign(N, 0.0);
    AS1.assign(N, 0.0);
    AP1.assign(N, 0.0);
#endif

    TCG.assign(N, 0.0);
    RES.assign(N, 0.0);
    PDIR.assign(N, 0.0);
    AP_CG.assign(N, 0.0);

    //32 in x so each warp is one contiguous run of 32 nodes along a row.
    //A 2-D grid avoids the runtime integer divide a 1-D map would need, since
    //NI and NJ are not compile-time constants here.
    BLK_INT = dim3(32, 8);
    GRD_INT = dim3(((NCELLI-1) + 31)/32, ((NCELLJ-1) + 7)/8);
    GRD_ALL = dim3((NI + 31)/32, (NJ + 7)/8);

    NBLK_INT = (int)(GRD_INT.x * GRD_INT.y);
    PARTIAL.assign((size_t)NBLK_INT, 0.0);
    PARTIAL_N.assign((size_t)NBLK_INT, 0.0);

    H_CG = CG_SCALARS();
    H_CG.RS_OLD = H_CG.RS_NEW = H_CG.PAP_CG = 0.0;
    H_CG.ALPHA_CG = H_CG.BETA_CG = H_CG.RRMS_CG = H_CG.DTDT = H_CG.NORM = 0.0;
    H_CG.TOTAL_CG_ITER = 0;
    H_CG.DONE_CG = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(CG, &H_CG, sizeof(CG_SCALARS)));

    const double MB = ((UNIFORM_COEFF ? 7.0 : 17.0)*(double)N*sizeof(double))
                    / (1024.0*1024.0);

    cout << "LAUNCH  : block " << BLK_INT.x << "x" << BLK_INT.y
         << "   interior grid " << GRD_INT.x << "x" << GRD_INT.y
         << "   partials " << NBLK_INT << endl;
    cout << "MEMORY  : " << fixed << setprecision(2) << MB << " MB on device" << endl;
}

void FREE_FIELDS()
{
    DESTROY_CG_GRAPH();

    //Release while the CUDA context is still alive. These containers are
    //globals, so their destructors would otherwise run after main() returns and
    //call cudaFree on a torn-down context.
    Field().swap(SP);
    Field().swap(T);          Field().swap(T_old_time);
    Field().swap(AW);         Field().swap(AE);       Field().swap(AS);
    Field().swap(AN);         Field().swap(AP);
    Field().swap(AW1);        Field().swap(AE1);      Field().swap(AS1);
    Field().swap(AN1);        Field().swap(AP1);
    Field().swap(TCG);        Field().swap(RES);
    Field().swap(PDIR);       Field().swap(AP_CG);
    Field().swap(PARTIAL);    Field().swap(PARTIAL_N);

    PinnedField().swap(T_HOST);
    HostField().swap(XCELL);
}

void FETCH_FIELD()
{
    CUDA_CHECK(cudaMemcpy(T_HOST.data(), RAW(T),
                          (size_t)NI*NJ*sizeof(double), cudaMemcpyDeviceToHost));
}

//Norm of an arbitrary field, honouring NORM_TYPE. Costs one extra sweep plus a
//D2H sync, so keep it out of the CG loop -- the loop gets its norm free from
//the fused reductions. PARTIAL_N is used, not PARTIAL: calling this mid-CG must
//not stomp the residual partials CALC_ALPHA_CG_K is about to read.
double CALC_NORM_FIELD(const Field& ARR)
{
    const double invN = 1.0/((double)(NI-2)*(double)(NJ-2));

    CALC_NORM_K<<<GRD_INT, BLK_INT>>>(RAW(ARR), RAW(PARTIAL_N));
    FINALIZE_NORM_K<<<1, NTHREAD_RED>>>(RAW(PARTIAL_N), NBLK_INT, invN);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyFromSymbol(&H_CG, CG, sizeof(CG_SCALARS)));
    return H_CG.NORM;
}

void SET_GEOMETRY()
{
    DELX = LX/(NI-1);
    DELY = LY/(NJ-1);

    for(int j=0; j<NJ; j++){
       for(int i=0;i<NI;i++){
        XCELL[IDX(0,j,i)] = DELX*i;
        XCELL[IDX(1,j,i)] = DELY*j;

       }
     }

    //The BC kernel needs NI/NJ/DELX/DELY and runs before CALC_COEFF_TRANSIENT,
    //so the parameter block has to be pushed here as well as there.
    PUSH_PARAMS();
}

void APPLYIC_CG()
{
    APPLYIC_CG_K<<<GRD_ALL, BLK_INT>>>(RAW(TCG), RAW(RES), RAW(PDIR),
                                       RAW(AP_CG), RAW(T_old_time));
    CUDA_CHECK(cudaGetLastError());
}

void APPLYBC_TEMP_CG()
{
    APPLYBC_TEMP_K<<<1, NTHREAD_RED>>>(RAW(TCG));
    CUDA_CHECK(cudaGetLastError());
}

void CALC_RESIDUAL_CG()
{
    const double invN = 1.0/((double)(NI-2)*(double)(NJ-2));

    CALC_RESIDUAL_CG_K<<<GRD_INT, BLK_INT>>>(RAW(AP), RAW(AW), RAW(AE), RAW(AS), RAW(AN),
                                             RAW(TCG), RAW(SP), RAW(RES), RAW(PDIR),
                                             RAW(PARTIAL), RAW(PARTIAL_N));

    CALC_RS_OLD_K<<<1, NTHREAD_RED>>>(RAW(PARTIAL), RAW(PARTIAL_N), NBLK_INT, invN, CCSS);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyFromSymbol(&H_CG, CG, sizeof(CG_SCALARS)));
    RS_OLD  = H_CG.RS_OLD;
    RRMS_CG = H_CG.RRMS_CG;
}

void SOLVER_CG()
{
    //Solves A x = b by Conjugate Gradient. Knows nothing about time -- the
    //physical timestep loop lives in main(), and T_old_time is FROZEN for this
    //entire routine.
    //
    //  r_0     = b - A x_0
    //  p_0     = r_0                          (both set in CALC_RESIDUAL_CG)
    //  alpha_k = (r_k . r_k)/(p_k . A p_k)
    //  x_{k+1} = x_k + alpha_k p_k
    //  r_{k+1} = r_k - alpha_k A p_k
    //  beta_k  = (r_{k+1} . r_{k+1})/(r_k . r_k)
    //  p_{k+1} = r_{k+1} + beta_k p_k
    //
    //The convergence test lives on the device as CG.DONE_CG and every kernel is
    //predicated on it, so the host fires a fixed chunk of iterations and reads
    //the flag once per chunk rather than twice per iteration. Iterations past
    //convergence inside a chunk are no-ops that touch no memory.

    if(H_CG.DONE_CG)
    {
        ITER_CG = 0;
        return;
    }

    const long long ITER_BEFORE = H_CG.TOTAL_CG_ITER;
    int LAUNCHED = 0;

    while(LAUNCHED < MAXITER_CG)
    {
#if USE_CG_GRAPH
        CUDA_CHECK(cudaGraphLaunch(E_CG, STREAM_CG));
#else
        for(int k=0; k<CG_CHUNK; k++) LAUNCH_CG_ITER(STREAM_CG);
#endif
        CUDA_CHECK(cudaMemcpyFromSymbolAsync(&H_CG, CG, sizeof(CG_SCALARS), 0,
                                             cudaMemcpyDeviceToHost, STREAM_CG));
        CUDA_CHECK(cudaStreamSynchronize(STREAM_CG));

        LAUNCHED += CG_CHUNK;

        if(VERBOSE)
        {
            cout << "ITER_CG = " << (int)(H_CG.TOTAL_CG_ITER - ITER_BEFORE)
                 << "   RRMS_CG = " << H_CG.RRMS_CG << endl;
        }

        if(H_CG.DONE_CG) break;
    }

    CUDA_CHECK(cudaGetLastError());

    ALPHA_CG = H_CG.ALPHA_CG;
    BETA_CG  = H_CG.BETA_CG;
    RS_OLD   = H_CG.RS_OLD;
    RS_NEW   = H_CG.RS_NEW;
    PAP_CG   = H_CG.PAP_CG;
    RRMS_CG  = H_CG.RRMS_CG;

    ITER_CG = (int)(H_CG.TOTAL_CG_ITER - ITER_BEFORE);

    if(!H_CG.DONE_CG)
    {
        cout << "WARNING: CG hit MAXITER_CG at timestep " << TIMESTEP
             << "   RRMS_CG = " << RRMS_CG << endl;
    }

    TOTAL_CG_ITER = (long)H_CG.TOTAL_CG_ITER;
}

void APPLYIC_TRANSIENT()
{
    APPLYIC_TRANSIENT_K<<<GRD_ALL, BLK_INT>>>(RAW(T), RAW(T_old_time));
    CUDA_CHECK(cudaGetLastError());
}

void APPLYBC_TEMP_TRANSIENT()
{
    //T and T_old_time are independent in the CPU loop bodies, so they split
    //cleanly into two launches of the same kernel.
    APPLYBC_TEMP_K<<<1, NTHREAD_RED>>>(RAW(T));
    APPLYBC_TEMP_K<<<1, NTHREAD_RED>>>(RAW(T_old_time));
    CUDA_CHECK(cudaGetLastError());
}

double FO_LIMIT()
{
    if(BETA >= 0.5)
    {
        return -1.0;
    }

    return 1.0/(2.0*(1.0-2.0*BETA));
}

void CHECK_STABILITY()
{
    double Fo_sum = Fo_x + Fo_y;
    double Fo_lim = FO_LIMIT();

    if(Fo_lim < 0.0)
    {
        cout << "STABIL  : Fo_x+Fo_y = " << fixed << setprecision(6) << Fo_sum
             << "   BETA = " << BETA << " >= 0.5 -> unconditionally stable"
             << " (dt is set by ACCURACY, not stability)" << endl;
        return;
    }

    double dt_max = Fo_lim/(alpha*(1.0/(DELX*DELX) + 1.0/(DELY*DELY)));

    cout << "STABIL  : Fo_x+Fo_y = " << fixed << setprecision(6) << Fo_sum
         << "   limit = " << Fo_lim
         << "   dt_max = " << scientific << setprecision(4) << dt_max << endl;

    if(Fo_sum > Fo_lim)
    {
        cout << "ERROR: scheme is UNSTABLE at BETA = " << BETA
             << " for this dt. Reduce deltaT below " << dt_max
             << ", coarsen the grid, or raise BETA to 0.5 or above." << endl;
        exit(1);
    }

}

void SET_DELTAT()
{
    double Fo_lim = FO_LIMIT();

    if(Fo_lim < 0.0)
    {
        deltaT = DT_ACCURACY;

    }
    else
    {
        double dt_max = Fo_lim/(alpha*(1.0/(DELX*DELX) + 1.0/(DELY*DELY)));
        deltaT = DT_SAFETY*dt_max;

        if(deltaT > DT_ACCURACY)
        {
            deltaT = DT_ACCURACY;
        }
    }

    cout << "SCHEME  : BETA = " << fixed << setprecision(4) << BETA << "   (";
    if(BETA < BETA_ZERO_TOL)        cout << "explicit / FTCS";
    else if(fabs(BETA-0.5) < 1e-12) cout << "Crank-Nicolson";
    else if(fabs(BETA-1.0) < 1e-12) cout << "fully implicit / backward Euler";
    else                            cout << "generalised theta";
    cout << ")" << endl;
}

void CALC_FOURIER()
{
    //Fo_x = alpha*dt/dx^2,  Fo_y = alpha*dt/dy^2.

    Fo_x = alpha*deltaT/(DELX*DELX);
    Fo_y = alpha*deltaT/(DELY*DELY);

    cout << "TIME    : deltaT = " << scientific << setprecision(4) << deltaT
         << "   Fo_x = " << Fo_x << "   Fo_y = " << Fo_y << endl;
}

//Builds BOTH coefficient sets for every BETA, including BETA = 0:
//  AP/AW/AE/AS/AN  act on level n+1 (subtractive), with AP  - (AW+AE+AS+AN) = 1
//  AP1/AW1/../AN1  act on level n   (additive),    with AP1 + (AW1+AE1+AS1+AN1) = 1
void CALC_COEFF_TRANSIENT()
{
    PUSH_PARAMS();      //deltaT is known by now

    H_UCO.AW = BETA*Fo_x;
    H_UCO.AE = BETA*Fo_x;
    H_UCO.AN = BETA*Fo_y;
    H_UCO.AS = BETA*Fo_y;

    H_UCO.AP = 1.0 + 2.0*BETA*(Fo_x + Fo_y);

    H_UCO.AW1 = (1.0-BETA)*Fo_x;
    H_UCO.AE1 = (1.0-BETA)*Fo_x;
    H_UCO.AN1 = (1.0-BETA)*Fo_y;
    H_UCO.AS1 = (1.0-BETA)*Fo_y;

    H_UCO.AP1 = 1.0 - 2.0*(1.0-BETA)*(Fo_x + Fo_y);

    CUDA_CHECK(cudaMemcpyToSymbol(UCO, &H_UCO, sizeof(UNIFORM_COEFFS)));

#if UNIFORM_COEFF == 0
    CALC_COEFF_TRANSIENT_K<<<GRD_INT, BLK_INT>>>(RAW(AW), RAW(AE), RAW(AS), RAW(AN),
                                                 RAW(AP), RAW(AW1), RAW(AE1), RAW(AS1),
                                                 RAW(AN1), RAW(AP1),
                                                 BETA, Fo_x, Fo_y);
    CUDA_CHECK(cudaGetLastError());
#endif

    if(H_UCO.AP1 < 0.0)
    {
        cout << "WARNING : AP1 = " << fixed << setprecision(4) << H_UCO.AP1
             << " < 0 violates Patankar's positive-coefficient rule. The scheme is"
             << " stable but the solution may oscillate; non-oscillatory behaviour"
             << " needs Fo_x + Fo_y <= 1/(2*(1-BETA))." << endl;
    }
}

void BUILD_RHS_TRANSIENT()
{
    BUILD_RHS_TRANSIENT_K<<<GRD_INT, BLK_INT>>>(RAW(AP1), RAW(AW1), RAW(AE1),
                                                RAW(AS1), RAW(AN1),
                                                RAW(T_old_time), RAW(SP));
    CUDA_CHECK(cudaGetLastError());
}

void SOLVER_TRANSIENT()
{
    BUILD_RHS_TRANSIENT();

    if(BETA < BETA_ZERO_TOL)
    {
        COPY_INTERIOR_K<<<GRD_INT, BLK_INT>>>(RAW(T), RAW(SP));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    APPLYIC_CG();

    APPLYBC_TEMP_CG();

    CALC_RESIDUAL_CG();

    SOLVER_CG();

    COPY_INTERIOR_K<<<GRD_INT, BLK_INT>>>(RAW(T), RAW(TCG));
    CUDA_CHECK(cudaGetLastError());
}

double UPDATE_TRANSIENT()
{
    const double invN = 1.0/((double)(NI-2)*(double)(NJ-2));

    UPDATE_TRANSIENT_K<<<GRD_INT, BLK_INT>>>(RAW(T), RAW(T_old_time), RAW(PARTIAL));
    CALC_DTDT_K<<<1, NTHREAD_RED>>>(RAW(PARTIAL), NBLK_INT, invN);
    CUDA_CHECK(cudaGetLastError());

    //One D2H sync per timestep, because main() prints and tests the return value.
    CUDA_CHECK(cudaMemcpyFromSymbol(&H_CG, CG, sizeof(CG_SCALARS)));

    //swap now lives at the TOP of the timestep loop in main() -- keeping it here
    //left T holding level n, so every written file lagged one step behind its label.
    return H_CG.DTDT;
}

void WRITE_FILE_TRANSIENT()
{
    //Files numbered by TIMESTEP so nothing is overwritten:
    //      temperature_00000.dat, temperature_00100.dat, ...
    //STRANDID + SOLUTIONTIME let Tecplot assemble the sequence into a single transient
    //zone that animates directly.
    char fname[512];
    snprintf(fname, sizeof(fname), OUTPUT_DIR "/temperature_%05d.dat", TIMESTEP);

    ofstream out(fname);
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << fname << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    out << "TITLE = \"2D Transient Heat Conduction (FDM)\"" << endl;
    out << "VARIABLES = \"X\", \"Y\", \"T\"" << endl;
    out << "ZONE T=\"t=" << simTime << "\""
        << ", I=" << NI << ", J=" << NJ
        << ", F=POINT"
        << ", STRANDID=1, SOLUTIONTIME=" << simTime << endl;

    for (int j = 0; j < NJ; j++)
    {
        for (int i = 0; i < NI; i++)
        {
            out << XCELL[IDX(0,j,i)] << " " << XCELL[IDX(1,j,i)] << " "
                << T_HOST[ID(j,i)] << endl;
        }
    }

    out.close();
}

void WRITE_FILE_TRANSIENT_VTK()
{
    //Same STRUCTURED_GRID reasoning; numbered so ParaView picks the files up as
    //a time series automatically.
    char fname[512];
    snprintf(fname, sizeof(fname), OUTPUT_DIR "/temperature_%05d.vtk", TIMESTEP);

    ofstream out(fname);
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << fname << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    out << "# vtk DataFile Version 3.0" << endl;
    out << "2D Transient Heat Conduction, t = " << simTime << endl;
    out << "ASCII" << endl;
    out << "DATASET STRUCTURED_GRID" << endl;
    out << "DIMENSIONS " << NI << " " << NJ << " " << 1 << endl;
    out << "POINTS " << NI*NJ << " double" << endl;

    for (int j = 0; j < NJ; j++)
    {
        for (int i = 0; i < NI; i++)
        {
            out << XCELL[IDX(0,j,i)] << " " << XCELL[IDX(1,j,i)] << " " << 0.0 << endl;
        }
    }

    out << "POINT_DATA " << NI*NJ << endl;
    out << "SCALARS Temperature double 1" << endl;
    out << "LOOKUP_TABLE default" << endl;

    for (int j = 0; j < NJ; j++)
    {
        for (int i = 0; i < NI; i++)
        {
            const size_t p = ID(j, i);
            out << T_HOST[p] << endl;
        }
    }

    out.close();
}

void MAKE_OUTPUT_DIR()
{
    //Path is relative to the CURRENT WORKING DIRECTORY, not to the binary.
    if(MAKE_DIR(OUTPUT_DIR) != 0 && errno != EEXIST)
    {
        cout << "ERROR: could not create output directory '" << OUTPUT_DIR
             << "' -- errno = " << errno << endl;
        exit(1);
    }

    cout << "OUTPUT  : " << OUTPUT_DIR << "/" << endl;
}
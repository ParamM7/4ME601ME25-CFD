#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <errno.h>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <chrono>

#if defined(__GNUC__) || defined(__clang__)
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
using Field = std::vector<double>;

int NX = 64;
int NY = 64;

inline size_t IDU(int j, int i) { return (size_t)j*(NX+1) + i;}
inline size_t IDV(int j, int i) { return (size_t)j*(NX+2) + i;}
inline size_t IDP(int j, int i) { return (size_t)j*(NX+2) + i;}

#define SCH_FOU 0
#define SCH_SOU 1
#define SCH_QUICK 2

int SCHEME = SCH_FOU;

const char* SCHEME_NAME()
{
    if(SCHEME == SCH_SOU) return "SOU";
    else if (SCHEME == SCH_QUICK) return "QUICK";
    else return "FOU";
}

int PARSE_SCHEME(const char* s)
{
    if(!s) return -1;
    if(!strcmp(s,"0") || !strcmp(s,"fou")   || !strcmp(s,"FOU"))   return SCH_FOU;
    if(!strcmp(s,"1") || !strcmp(s,"sou")   || !strcmp(s,"SOU"))   return SCH_SOU;
    if(!strcmp(s,"2") || !strcmp(s,"quick") || !strcmp(s,"QUICK")) return SCH_QUICK;
    return -1;
}

Field U, V, P;
Field U_old, V_old;
Field US, VS;
Field DIV;

Field AW, AE, AN, AS, AP;
Field SPBC;
Field SP;
Field RES, PDIR, AP_CG;
Field RESN;

void ALLOCATE_FIELDS();

double DELX, DELY;
double AFX, AFY;
double VOL;

double LX, LY;
double CCSS;

bool VERBOSE = false;

#define NORM_L1 1
#define NORM_L2 2

#define NORM_TYPE NORM_L2

#if NORM_TYPE == NORM_L1
    #define CALC_NORM CALC_NORM_L1
#else
    #define CALC_NORM CALC_NORM_L2
#endif

double CALC_NORM_L2(const Field& ARR)
{
    int ii, jj;

    double sumsq = 0.0;

    for(jj=1;jj<=NY;jj++)
    {
        for(ii=1;ii<=NX;ii++)
        {
            const double v = ARR[IDP(jj,ii)];
            sumsq = sumsq + v*v;
        }
    }

    return sqrt(sumsq/((double)NX*(double)NY));
}

double CALC_NORM_L1(const Field& ARR)
{
    int ii, jj;

    double sumabs = 0.0;

    for(jj=1;jj<=NY;jj++)
    {
        for(ii=1;ii<=NX;ii++)
        {
            sumabs = sumabs + fabs(ARR[IDP(jj,ii)]);
        }
    }

    return sumabs/((double)NX*(double)NY);
}

//solver control parameters

#define rho 1.0
#define U_LID 1.0

double RE = 100.0;
double nu;

double simTime;
double deltaT;
int TIMESTEP;

#define CFL_TARGET 0.5
#define DT_ACCURACY 0.01
#define DT_FLOOR 1.0e-8
#define TMAX 150.0
#define VN_SAFETY 0.8
#define MAXSTEP 2000000

double WRITE_INTERVAL = 0.5;
double nextWriteTime = WRITE_INTERVAL;
double io_s = 0.0;

#define OUTPUT_DIR "results"

//conjugate gradient solver parameters

#define MAXITER_CG 50000
#define RTOL_CG 1.0e-8

double ALPHA_CG;
double BETA_CG;
double RS_OLD, RS_NEW;
double RRMS_CG;
double PAP_CG;
int ITER_CG;
long TOTAL_CG_ITER = 0;

//pressure pinning parameters for LDC

int PIN_I = 1;
int PIN_J = 1;
bool PIN_ACTIVE = true;

double WORST_DIV = 0.0;

//boundary condition table

//u
#define AU_WEST 0.0
#define BU_WEST 1.0
#define CU_WEST 0.0

#define AU_EAST 0.0
#define BU_EAST 1.0
#define CU_EAST 0.0

#define AU_SOUTH 0.0
#define BU_SOUTH 1.0
#define CU_SOUTH 0.0

#define AU_NORTH 0.0
#define BU_NORTH 1.0
#define CU_NORTH U_LID

//v
#define AV_WEST 0.0
#define BV_WEST 1.0
#define CV_WEST 0.0

#define AV_EAST 0.0
#define BV_EAST 1.0
#define CV_EAST 0.0

#define AV_SOUTH 0.0
#define BV_SOUTH 1.0
#define CV_SOUTH 0.0

#define AV_NORTH 0.0
#define BV_NORTH 1.0
#define CV_NORTH 0.0

//p
#define AP_WEST 1.0
#define BP_WEST 0.0
#define CP_WEST 0.0

#define AP_EAST 1.0
#define BP_EAST 0.0
#define CP_EAST 0.0

#define AP_SOUTH 1.0
#define BP_SOUTH 0.0
#define CP_SOUTH 0.0

#define AP_NORTH 1.0
#define BP_NORTH 0.0
#define CP_NORTH 0.0

double BC_A_W[3], BC_B_W[3], BC_C_W[3];
double BC_A_S[3], BC_B_S[3], BC_C_S[3];
double BC_A_E[3], BC_B_E[3], BC_C_E[3];
double BC_A_N[3], BC_B_N[3], BC_C_N[3];

void SET_GEOMETRY();
void SET_BC_COEFF();
void VALIDATE_BC();
void MAKE_OUTPUT_DIR();
void APPLYIC();
void APPLYBC_U(Field& PHI);
void APPLYBC_V(Field& PHI);
void APPLYBC_P(Field& PHI);
void SET_DELTAT();
double CALC_CONV_U(int j, int i);
double CALC_CONV_V(int j, int i);
double CALC_DIFF_U(int j, int i);
double CALC_DIFF_V(int j, int i);
void PREDICTOR();
void CALC_DIVERGENCE(const Field& UU, const Field& VV);
void CALC_COEFF_P();
void BUILD_RHS_P();
void APPLYIC_CG();
void CALC_RESIDUAL_CG();
void SOLVER_CG();
void SOLVER_PRESSURE();
void CORRECTOR();
double UPDATE_TRANSIENT();
void CHECK_CONTINUITY();
void CHECK_MASS_BALANCE();
void WRITE_FILE_TRANSIENT();
void WRITE_FILE_TRANSIENT_VTK();

//main

int main(int argc, char* argv[])
{
    const auto t_wall_start = std::chrono::steady_clock::now();

    CCSS = 1.0e-12;
    LX = 1.0;
    LY = 1.0;

    if(argc > 1)
    {
        SCHEME = PARSE_SCHEME(argv[1]);
        if(SCHEME < 0)
        {
            cout << "usage: " <<  argv[0] << " [fou|sou|quick] [NX NY]" << endl;
            return -1;
        }
    }
    else
    {
        int s = -1;
        cout << "CONVECTION SCHEME: [0] FOU, [1] SOU, [2] QUICK -> ";
        if(!(cin >> s) || s < 0 || s > 2)
        {
            cout << endl << "no valid scheme selected, defaulting to FOU" << endl;
            s = SCH_FOU;
        }
        SCHEME = s;
    }

    if(argc > 3)
    {
        NX = atoi(argv[2]);
        NY = atoi(argv[3]);

        if (NX < 5 || NY < 5)
        {
            cout << "ERROR: NX and NY must be >= 5" << endl;
            return -1;
        }
    }

    if(argc > 4)
    {
        RE = atof(argv[4]);
        if(RE <= 0.0)
        {
            cout << "ERROR: RE must be > 0" << endl;
            return -1;
        }
    }

    nu = U_LID*LX/RE;

    PIN_I = 1;
    PIN_J = 1;

    cout << "GRID    : NX = " << NX << "   NY = " << NY
         << "   -> " << NX << " x " << NY << " pressure cells" << endl;
    cout << "          (U is " << NX+1 << " x " << NY+2
         << ",  V is " << NX+2 << " x " << NY+1
         << ",  P is " << NX+2 << " x " << NY+2 << ", staggered MAC)" << endl;
    cout << "SCHEME  : " << SCHEME_NAME() << endl;
    cout << "METHOD  : Chorin projection  ->  U* = U^n + dt*(-conv+diff),"
         << "  lap(p) = (rho/dt)*div(U*),  U^{n+1} = U* - (dt/rho)*grad(p)" << endl;
    cout << "PHYSICS : Re = " << RE << "   nu = " << scientific << setprecision(4)
         << nu << "   U_LID = " << U_LID << endl;

    ALLOCATE_FIELDS();
    MAKE_OUTPUT_DIR();
    SET_GEOMETRY();
    SET_BC_COEFF();

    APPLYIC();

    APPLYBC_U(U);
    APPLYBC_V(V);
    APPLYBC_P(P);

    CALC_COEFF_P();

    TIMESTEP = 0;
    simTime = 0.0;

    WRITE_FILE_TRANSIENT();
    WRITE_FILE_TRANSIENT_VTK();

    while(simTime < TMAX && TIMESTEP <MAXSTEP)
    {
        std::swap(U, U_old);
        std::swap(V, V_old);

        SET_DELTAT();

        PREDICTOR();
        APPLYBC_U(US);
        APPLYBC_V(VS);

        CALC_DIVERGENCE(US, VS);
        SOLVER_PRESSURE();
        APPLYBC_P(P);

        CORRECTOR();
        APPLYBC_U(U);
        APPLYBC_V(V);

        CHECK_CONTINUITY();

        const double dUdt = UPDATE_TRANSIENT();

        simTime += deltaT;
        TIMESTEP++;

        if(TIMESTEP % 10 == 0)
        {
            cout << "STEP = " << setw(6) << TIMESTEP
                 << "   t = "  << fixed << setprecision(6) << simTime
                 << "   dt = " << scientific << setprecision(3) << deltaT
                 << "   ||dU/dt|| = " << setprecision(4) << dUdt
                 << "   max|div| = " << WORST_DIV
                 << "   CG/step = " << TOTAL_CG_ITER/(long)TIMESTEP
                 << endl;
        }

        if(simTime + 0.5*deltaT >= nextWriteTime)
        {
            const auto t_io0 = std::chrono::steady_clock::now();

            WRITE_FILE_TRANSIENT();
            WRITE_FILE_TRANSIENT_VTK();

            io_s += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_io0).count();

            nextWriteTime += WRITE_INTERVAL;
        }
    }

    WRITE_FILE_TRANSIENT();
    WRITE_FILE_TRANSIENT_VTK();

    CHECK_MASS_BALANCE();

    const auto t_wall_end = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t_wall_end - t_wall_start).count();

    cout << "Total CG iterations = " << TOTAL_CG_ITER
         << "   (average " << (double)TOTAL_CG_ITER/(double)(TIMESTEP>0?TIMESTEP:1)
         << " per timestep, 1 Poisson solve per step)" << endl;

    cout << "WALLTIME: total = " << fixed << setprecision(3) << wall_s << " s"
         << "   I/O = " << io_s << " s"
         << "   compute = " << wall_s - io_s << " s" << endl;
    cout << "          (" << TIMESTEP << " steps, "
         << scientific << setprecision(3)
         << (double)TIMESTEP*(double)NX*(double)NY/((wall_s - io_s) > 0.0 ? (wall_s - io_s) : 1.0)
         << " cell-updates/s compute-only)" << endl;

    return 0;
}


//setup

void ALLOCATE_FIELDS()
{
    const size_t NU = (size_t)(NX+1)*(size_t)(NY+2);
    const size_t NV = (size_t)(NX+2)*(size_t)(NY+1);
    const size_t NP = (size_t)(NX+2)*(size_t)(NY+2);

    U.assign(NU, 0.0); U_old.assign(NU, 0.0); US.assign(NU, 0.0);
    V.assign(NV, 0.0); V_old.assign(NV, 0.0); VS.assign(NV, 0.0);

    P.assign(NP, 0.0);
    DIV.assign(NP, 0.0);

    AW.assign(NP, 0.0); AE.assign(NP, 0.0);
    AN.assign(NP, 0.0); AS.assign(NP, 0.0); AP.assign(NP, 0.0);
    SPBC.assign(NP, 0.0);

    SP.assign(NP, 0.0);
    RES.assign(NP, 0.0);
    AP_CG.assign(NP, 0.0);
    RESN.assign(NP, 0.0);
    PDIR.assign(NP, 0.0);
}
void SET_GEOMETRY()
{
    DELX = LX/(double)NX;
    DELY = LY/(double)NY;

    AFX = DELY;
    AFY = DELX;
    VOL = DELX*DELY;
}

void SET_BC_COEFF()
{
    BC_A_W[0] = AU_WEST; BC_B_W[0] = BU_WEST; BC_C_W[0] = CU_WEST;
    BC_A_E[0] = AU_EAST; BC_B_E[0] = BU_EAST; BC_C_E[0] = CU_EAST;
    BC_A_S[0] = AU_SOUTH; BC_B_S[0] = BU_SOUTH; BC_C_S[0] = CU_SOUTH;
    BC_A_N[0] = AU_NORTH; BC_B_N[0] = BU_NORTH; BC_C_N[0] = CU_NORTH;

    BC_A_W[1] = AV_WEST; BC_B_W[1] = BV_WEST; BC_C_W[1] = CV_WEST;
    BC_A_E[1] = AV_EAST; BC_B_E[1] = BV_EAST; BC_C_E[1] = CV_EAST;
    BC_A_S[1] = AV_SOUTH; BC_B_S[1] = BV_SOUTH; BC_C_S[1] = CV_SOUTH;
    BC_A_N[1] = AV_NORTH; BC_B_N[1] = BV_NORTH; BC_C_N[1] = CV_NORTH;

    BC_A_W[2] = AP_WEST; BC_B_W[2] = BP_WEST; BC_C_W[2] = CP_WEST;
    BC_A_E[2] = AP_EAST; BC_B_E[2] = BP_EAST; BC_C_E[2] = CP_EAST;
    BC_A_S[2] = AP_SOUTH; BC_B_S[2] = BP_SOUTH; BC_C_S[2] = CP_SOUTH;
    BC_A_N[2] = AP_NORTH; BC_B_N[2] = BP_NORTH; BC_C_N[2] = CP_NORTH;

    VALIDATE_BC();
}

void VALIDATE_BC()
{
    const char* wall[4] = {"WEST","SOUTH","EAST","NORTH"};
    const char* comp[3] = {"u","v","p"};

    for(int c=0; c<3; c++)
    {
        // One row per wall: a, b, and the wall spacing that goes with it.
        const double a[4] = {BC_A_W[c], BC_A_S[c], BC_A_E[c], BC_A_N[c]};
        const double b[4] = {BC_B_W[c], BC_B_S[c], BC_B_E[c], BC_B_N[c]};
        const double d[4] = {DELX, DELY, DELX, DELY};

        for(int w=0; w<4; w++)
        {
            // Direct (wall-normal) form divides by a + b*delta; ghost
            // (wall-tangential and pressure) form divides by b/2 - a/delta.
            const double dirden = a[w] + b[w]*d[w];
            const double ghoden = 0.5*b[w] - a[w]/d[w];

            if(fabs(dirden) < 1.0e-30 || fabs(ghoden) < 1.0e-30)
            {
                cout << "ERROR: " << wall[w] << " boundary condition on " << comp[c]
                     << " is degenerate -- a and b cannot both vanish, and"
                     << " a pure-Neumann tangential/pressure wall needs a != 0."
                     << endl;
                exit(1);
            }
        }
    }

    // A pressure field determined only by its normal derivative is fixed only up
    // to an additive constant; the Poisson matrix is then singular and CG needs
    // one degree of freedom removed.
    PIN_ACTIVE = (fabs(BC_B_W[2]) < 1.0e-30 && fabs(BC_B_E[2]) < 1.0e-30 &&
                  fabs(BC_B_S[2]) < 1.0e-30 && fabs(BC_B_N[2]) < 1.0e-30);

    if(PIN_ACTIVE)
    {
        cout << "NOTE    : p is Neumann on all four walls -- Poisson matrix is"
             << " singular, pinning P[" << PIN_J << "][" << PIN_I << "] = 0."
             << endl;
    }

    if(fabs(BC_B_W[0]) < 1.0e-30 || fabs(BC_B_E[0]) < 1.0e-30 ||
       fabs(BC_B_S[1]) < 1.0e-30 || fabs(BC_B_N[1]) < 1.0e-30)
    {
        cout << "NOTE    : at least one wall is zero-gradient in its NORMAL"
             << " component -- the domain is permeable there, and the pure-Neumann"
             << " pressure problem may no longer be compatible." << endl;
    }
}

void MAKE_OUTPUT_DIR()
{
    if(MAKE_DIR(OUTPUT_DIR) != 0 && errno != EEXIST)
    {
        cout << "ERROR: could not create output directory '" << OUTPUT_DIR
             << "' -- errno = " << errno << endl;
        exit(1);
    }

    cout << "OUTPUT  : " << OUTPUT_DIR << "/" << endl;
}

void APPLYIC()
{
    for(size_t p=0; p<U.size(); p++)
    {
        U[p] = 0.0;
        U_old[p] = 0.0;
        US[p] = 0.0;
    }
    for(size_t p=0; p<V.size(); p++)
    {
        V[p] = 0.0;
        V_old[p] = 0.0;
        VS[p] = 0.0;
    }
    for(size_t p=0; p<P.size(); p++)
    {
        P[p] = 0.0;
    }
}

inline double WALL_VALUE(double a, double b, double c, double delta, double phi_int)
{
    return (a*phi_int + c*delta)/(a + b*delta);
}

inline double GHOST_SLOPE(double a, double b, double delta)
{
    return -(a/delta + 0.5*b)/(0.5*b - a/delta);
}

inline double GHOST_OFFSET(double a, double b, double c, double delta)
{
    return c/(0.5*b - a/delta);
}

inline double GHOST_VALUE(double a, double b, double c, double delta, double phi_int)
{
    return GHOST_SLOPE(a,b,delta)*phi_int + GHOST_OFFSET(a,b,c,delta);
}

void APPLYBC_U(Field& PHI)
{
    const double aW = BC_A_W[0], bW = BC_B_W[0], cW = BC_C_W[0];
    const double aE = BC_A_E[0], bE = BC_B_E[0], cE = BC_C_E[0];
    const double aS = BC_A_S[0], bS = BC_B_S[0], cS = BC_C_S[0];
    const double aN = BC_A_N[0], bN = BC_B_N[0], cN = BC_C_N[0];

    for(int j=0; j<=NY+1; j++)
    {
        PHI[IDU(j,0)]  = WALL_VALUE(aW,bW,cW,DELX,PHI[IDU(j,1)]);
        PHI[IDU(j,NX)] = WALL_VALUE(aE,bE,cE,DELX,PHI[IDU(j,NX-1)]);
    }

    for(int i=0; i<=NX; i++)
    {
        PHI[IDU(0,i)]    = GHOST_VALUE(aS,bS,cS,DELY,PHI[IDU(1,i)]);
        PHI[IDU(NY+1,i)] = GHOST_VALUE(aN,bN,cN,DELY,PHI[IDU(NY,i)]);
    }
}

void APPLYBC_V(Field& PHI)
{
    const double aW = BC_A_W[1], bW = BC_B_W[1], cW = BC_C_W[1];
    const double aE = BC_A_E[1], bE = BC_B_E[1], cE = BC_C_E[1];
    const double aS = BC_A_S[1], bS = BC_B_S[1], cS = BC_C_S[1];
    const double aN = BC_A_N[1], bN = BC_B_N[1], cN = BC_C_N[1];

    // south and north walls: v sits ON them
    for(int i=0; i<=NX+1; i++)
    {
        PHI[IDV(0,i)]  = WALL_VALUE(aS, bS, cS, DELY, PHI[IDV(1,i)]);
        PHI[IDV(NY,i)] = WALL_VALUE(aN, bN, cN, DELY, PHI[IDV(NY-1,i)]);
    }

    // west and east ghost columns
    for(int j=0; j<=NY; j++)
    {
        PHI[IDV(j,0)]    = GHOST_VALUE(aW, bW, cW, DELX, PHI[IDV(j,1)]);
        PHI[IDV(j,NX+1)] = GHOST_VALUE(aE, bE, cE, DELX, PHI[IDV(j,NX)]);
    }
}

void APPLYBC_P(Field& PHI)
{
    const double aW = BC_A_W[2], bW = BC_B_W[2], cW = BC_C_W[2];
    const double aE = BC_A_E[2], bE = BC_B_E[2], cE = BC_C_E[2];
    const double aS = BC_A_S[2], bS = BC_B_S[2], cS = BC_C_S[2];
    const double aN = BC_A_N[2], bN = BC_B_N[2], cN = BC_C_N[2];

    for(int j=1; j<=NY; j++)
    {
        PHI[IDP(j,0)] = GHOST_VALUE(aW, bW, cW, DELX, PHI[IDP(j,1)]);
        PHI[IDP(j,NX+1)] = GHOST_VALUE(aE, bE, cE, DELX, PHI[IDP(j,NX)]);
    }

    for(int i=1; i<=NX; i++)
    {
        PHI[IDP(0,i)] = GHOST_VALUE(aS, bS, cS, DELY, PHI[IDP(1,i)]);
        PHI[IDP(NY+1,i)] = GHOST_VALUE(aN, bN, cN, DELY, PHI[IDP(NY,i)]);
    }

    PHI[IDP(0,0)] = 0.5*(PHI[IDP(1,0)] + PHI[IDP(0,1)]);
    PHI[IDP(0,NX+1)] = 0.5*(PHI[IDP(1,NX+1)] + PHI[IDP(0,NX)]);
    PHI[IDP(NY+1,0)] = 0.5*(PHI[IDP(NY,0)] + PHI[IDP(NY+1,1)]);
    PHI[IDP(NY+1,NX+1)] = 0.5*(PHI[IDP(NY,NX+1)] + PHI[IDP(NY+1,NX)]);
}

inline double V_AT_U (const Field& VV, int j, int i)
{
    return 0.25*(VV[IDV(j,i)] + VV[IDV(j, i+1)] + VV[IDV(j-1, i)] + VV[IDV(j-1, i+1)]);
}

inline double U_AT_V (const Field& UU, int j, int i)
{
    return 0.25*(UU[IDU(j,i)] + UU[IDU(j+1, i)] + UU[IDU(j, i-1)] + UU[IDU(j+1, i-1)]);
}

void SET_DELTAT()
{
    double cmax = 0.0;

    for(int j=1; j<=NY; j++)
    {
        for(int i=0; i<=NX; i++)
        {
            const double uu = fabs(U_old[IDU(j,i)]);
            const double vv = fabs(V_AT_U(V_old,j,i));
            cmax = max(cmax, uu/DELX + vv/DELY);
        }
    }

    for(int j=0; j<=NY; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            const double uu = fabs(U_AT_V(U_old,j,i));
            const double vv = fabs(V_old[IDV(j,i)]);
            cmax = max(cmax, uu/DELX + vv/DELY);
        }
    }

    double dt = DT_ACCURACY;

    if(cmax > 1.0e-12)
    {
        dt = min(dt, CFL_TARGET/cmax);
    }

    const double dt_visc = VN_SAFETY*0.5/(nu*(1.0/(DELX*DELX) + 1.0/(DELY*DELY)));
    dt = min(dt, dt_visc);

    deltaT = max(dt, DT_FLOOR);

    if(simTime + deltaT > TMAX) deltaT = TMAX - simTime;
}

//predictor

inline void FACE_WEIGHTS(double& wU, double& wD, double& wUU)
{
    if(SCHEME == SCH_SOU) { wU = 1.500; wD = 0.000; wUU = -0.500; return; }
    if(SCHEME == SCH_QUICK) { wU = 0.750; wD = 0.375; wUU = -0.125; return; }

    wU = 1.000; wD = 0.000; wUU = 0.000;
}

inline double FACE_VALUE(double F, double PHI_MM, double PHI_M, double PHI_P, double PHI_PP)
{
    double wU, wD, wUU;
    FACE_WEIGHTS(wU, wD, wUU);

    if(F >= 0.0) return wU*PHI_M + wD*PHI_P + wUU*PHI_MM;
    else return wU*PHI_P + wD*PHI_M + wUU*PHI_PP;
}

double CALC_CONV_U(int j, int i)
{
    const double* RESTRICT u = U_old.data();
    const double* RESTRICT v = V_old.data();

    const double uP = u[IDU(j,i)];
    const double uW = u[IDU(j,i-1)];
    const double uE = u[IDU(j,i+1)];
    const double uS = u[IDU(j-1,i)];
    const double uN = u[IDU(j+1,i)];

    const double uWW = (i-2 >= 0) ? u[IDU(j,i-2)] : uW;
    const double uEE = (i+2 <= NX) ? u[IDU(j,i+2)] : uE;
    const double uSS = (j-2 >= 0) ? u[IDU(j-2,i)] : uS;
    const double uNN = (j+2 <= NY+1) ? u[IDU(j+2,i)] : uN;

    const double Fe = 0.5*(uP + uE);
    const double Fw = 0.5*(uW + uP);

    const double ue = FACE_VALUE(Fe, uW,  uP, uE, uEE);
    const double uw = FACE_VALUE(Fw, uWW, uW, uP, uE);

    const double Fn = 0.5*(v[IDV(j,i)] + v[IDV(j,i+1)]);
    const double Fs = 0.5*(v[IDV(j-1,i)] + v[IDV(j-1,i+1)]);

    const double un = FACE_VALUE(Fn, uS, uP, uN, uNN);
    const double us = FACE_VALUE(Fs, uSS, uS, uP, uN);

    return (Fe*ue-Fw*uw)/DELX + (Fn*un-Fs*us)/DELY;
}

double CALC_DIFF_U(int j, int i)
{
    const double* RESTRICT u = U_old.data();

    const double uP = u[IDU(j,i)];
    const double uW = u[IDU(j,i-1)];
    const double uE = u[IDU(j,i+1)];
    const double uS = u[IDU(j-1,i)];
    const double uN = u[IDU(j+1,i)];

    const double DW = AFX/DELX;
    const double DE = AFX/DELX;
    const double DS = AFY/DELY;
    const double DN = AFY/DELY;

    return (DE*(uE-uP) - DW*(uP-uW) + (DN*(uN-uP) - DS*(uP-uS)))/VOL;
}

double CALC_CONV_V(int j, int i)
{
    const double* RESTRICT u = U_old.data();
    const double* RESTRICT v = V_old.data();

    const double vP = v[IDV(j,i)];
    const double vW = v[IDV(j,i-1)];
    const double vE = v[IDV(j,i+1)];
    const double vS = v[IDV(j-1,i)];
    const double vN = v[IDV(j+1,i)];

    const double vWW = (i-2 >= 0)     ? v[IDV(j,i-2)] : vW;
    const double vEE = (i+2 <= NX+1)  ? v[IDV(j,i+2)] : vE;
    const double vSS = (j-2 >= 0)     ? v[IDV(j-2,i)] : vS;
    const double vNN = (j+2 <= NY)    ? v[IDV(j+2,i)] : vN;

    const double Fn = 0.5*(vP + vN);
    const double Fs = 0.5*(vS + vP);

    const double vn = FACE_VALUE(Fn, vS,  vP, vN, vNN);
    const double vs = FACE_VALUE(Fs, vSS, vS, vP, vN);

    const double Fe = 0.5*(u[IDU(j,i)]   + u[IDU(j+1,i)]);
    const double Fw = 0.5*(u[IDU(j,i-1)] + u[IDU(j+1,i-1)]);

    const double ve = FACE_VALUE(Fe, vW,  vP, vE, vEE);
    const double vw = FACE_VALUE(Fw, vWW, vW, vP, vE);

    return (Fe*ve - Fw*vw)/DELX + (Fn*vn - Fs*vs)/DELY;
}

double CALC_DIFF_V(int j, int i)
{
    const double* RESTRICT v = V_old.data();

    const double vP = v[IDV(j,i)];
    const double vW = v[IDV(j,i-1)];
    const double vE = v[IDV(j,i+1)];
    const double vS = v[IDV(j-1,i)];
    const double vN = v[IDV(j+1,i)];

    const double DW = AFX/DELX;
    const double DE = AFX/DELX;
    const double DS = AFY/DELY;
    const double DN = AFY/DELY;

    return ( DE*(vE - vP) - DW*(vP - vW) + DN*(vN - vP) - DS*(vP - vS) ) / VOL;
}

void PREDICTOR()
{
    double* RESTRICT us = US.data();
    double* RESTRICT vs = VS.data();
    const double* RESTRICT uo = U_old.data();
    const double* RESTRICT vo = V_old.data();

    for(int j=1; j<=NY; j++)
    {
        for(int i=1; i<=NX-1; i++)
        {
            const size_t p = IDU(j,i);
            us[p] = uo[p] + deltaT*(-CALC_CONV_U(j,i) + nu*CALC_DIFF_U(j,i));
        }
    }

    for(int j=1; j<=NY-1; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            const size_t p = IDV(j,i);
            vs[p] = vo[p] + deltaT*(-CALC_CONV_V(j,i) + nu*CALC_DIFF_V(j,i));
        }
    }
}

void CALC_DIVERGENCE(const Field& UU, const Field& VV)
{
    const double* RESTRICT uu= UU.data();
    const double* RESTRICT vv = VV.data();
    double * RESTRICT d = DIV.data();

    for(int j=1; j<=NY; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            d[IDP(j,i)] = (uu[IDU(j,i)] - uu[IDU(j,i-1)])/DELX
                        + (vv[IDV(j,i)] - vv[IDV(j-1,i)])/DELY;
        }
    }
}

void CALC_COEFF_P()
{
    const double axx = 1.0/(DELX*DELX);
    const double ayy = 1.0/(DELY*DELY);

    for(int j=1; j<=NY; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            const size_t p = IDP(j,i);

            double ap  = 0.0;
            double sbc = 0.0;

            // west
            if(i > 1) { AW[p] = axx; ap += axx; }
            else
            {
                AW[p] = 0.0;
                ap  += axx*(1.0 - GHOST_SLOPE(BC_A_W[2], BC_B_W[2], DELX));
                sbc += axx*GHOST_OFFSET(BC_A_W[2], BC_B_W[2], BC_C_W[2], DELX);
            }

            // east
            if(i < NX) { AE[p] = axx; ap += axx; }
            else
            {
                AE[p] = 0.0;
                ap  += axx*(1.0 - GHOST_SLOPE(BC_A_E[2], BC_B_E[2], DELX));
                sbc += axx*GHOST_OFFSET(BC_A_E[2], BC_B_E[2], BC_C_E[2], DELX);
            }

            // south
            if(j > 1) { AS[p] = ayy; ap += ayy; }
            else
            {
                AS[p] = 0.0;
                ap  += ayy*(1.0 - GHOST_SLOPE(BC_A_S[2], BC_B_S[2], DELY));
                sbc += ayy*GHOST_OFFSET(BC_A_S[2], BC_B_S[2], BC_C_S[2], DELY);
            }

            // north
            if(j < NY) { AN[p] = ayy; ap += ayy; }
            else
            {
                AN[p] = 0.0;
                ap  += ayy*(1.0 - GHOST_SLOPE(BC_A_N[2], BC_B_N[2], DELY));
                sbc += ayy*GHOST_OFFSET(BC_A_N[2], BC_B_N[2], BC_C_N[2], DELY);
            }

            AP[p]   = ap;
            SPBC[p] = sbc;
        }
    }
}

void BUILD_RHS_P()
{
    const double fac = -rho/deltaT;

    for(int j=1; j<=NY; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            const size_t p = IDP(j,i);
            SP[p] = fac*DIV[p] + SPBC[p];
        }
    }
}

void APPLYIC_CG()
{
    for(size_t p=0; p<RES.size(); p++)
    {
        RES[p] = 0.0;
        AP_CG[p] = 0.0;
        PDIR[p] = 0.0;
    }
}

void CALC_RESIDUAL_CG()
{
    const int np = NX + 2;

    const double* RESTRICT ap = AP.data();
    const double* RESTRICT aw = AW.data();
    const double* RESTRICT ae = AE.data();
    const double* RESTRICT as = AS.data();
    const double* RESTRICT an = AN.data();
    const double* RESTRICT sp = SP.data();
    double* RESTRICT res = RES.data();
    double* RESTRICT pcg = P.data();
    double* RESTRICT pdir = PDIR.data();

    const size_t ppin = IDP(PIN_J, PIN_I);

    if(PIN_ACTIVE) pcg[ppin] = 0.0;

    double rs = 0.0;

    for(int j=1;j<=NY;j++)
    {
        for(int i=1;i<=NX; i++)
        {
            const size_t p= IDP(j,i);

            if(PIN_ACTIVE && p == ppin)
            {
                res[p] = 0.0;
                pdir[p] = 0.0;
                continue;
            }

            const double Ax = ap[p]*pcg[p] - aw[p]*pcg[p-1] - ae[p]*pcg[p+1] - as[p]*pcg[p-np] - an[p]*pcg[p+np];

            res[p] = sp[p]-Ax;
            pdir[p] = res[p];
            rs += res[p]*res[p];
        }
    }
    RS_OLD = rs;
}

void SOLVER_CG()
{
    const int np=NX+2;
    const double invN = 1.0/((double)NX*(double)NY);

    const double* RESTRICT ap = AP.data();
    const double* RESTRICT aw = AW.data();
    const double* RESTRICT ae = AE.data();
    const double* RESTRICT as = AS.data();
    const double* RESTRICT an = AN.data();
    double* RESTRICT res = RES.data();
    double* RESTRICT pcg = P.data();
    double* RESTRICT pd = PDIR.data();
    double* RESTRICT apcg = AP_CG.data();

    const size_t ppin = IDP(PIN_J, PIN_I);

    RRMS_CG = sqrt(RS_OLD*invN);

    const double RRSM0 = RRMS_CG;
    const double target = max(CCSS, RTOL_CG*RRSM0);

    if(RRMS_CG < target)
    {
        ITER_CG = 0;
        return;
    }

    for(ITER_CG=1; ITER_CG<=MAXITER_CG; ITER_CG++)
    {
        double pap = 0.0;

        for(int j=1; j<=NY; j++)
        {
            const size_t row = (size_t)j*np;

            for(int i=1; i<=NX; i++)
            {
                const size_t p = row + i;

                if(PIN_ACTIVE && p == ppin) { apcg[p] =0.0; continue;}

                apcg[p] = ap[p]*pd[p] - aw[p]*pd[p-1] - ae[p]*pd[p+1] - as[p]*pd[p-np] - an[p]*pd[p+np];

                pap += pd[p]*apcg[p];
            }
        }

        PAP_CG = pap;

        if(fabs(PAP_CG) < 1.0e-30)
        {
            break;
        }

        ALPHA_CG = RS_OLD/PAP_CG;

        double rsnew = 0.0;

        for(int j=1;j<=NY; j++)
        {
            const size_t row = (size_t)j*np;

            for(int i=1; i<=NX; i++)
            {
                const size_t p = row + i;

                pcg[p] += ALPHA_CG*pd[p];
                res[p] -= ALPHA_CG*apcg[p];

                rsnew += res[p]*res[p];
            }
        }

        RS_NEW = rsnew;

        RRMS_CG = sqrt(RS_NEW*invN);

        if(RRMS_CG < target)
        {
            break;
        }

        BETA_CG = RS_NEW/RS_OLD;

        for(int j=1; j<=NY; j++)
        {
            const size_t row = (size_t)j*np;

            for(int i=1; i<=NX; i++)
            {
                const size_t p = row + i;

                pd[p] = res[p] + BETA_CG*pd[p];
            }
        }

        RS_OLD = RS_NEW;

        if(VERBOSE && ITER_CG % 10 == 0)
        {
            cout << "ITER_CG = " << ITER_CG << "   RRMS_CG = " << RRMS_CG << endl;
        }
    }

    if(ITER_CG > MAXITER_CG)
    {
        cout << "WARNING: CG solver did not converge in " << MAXITER_CG
             << " iterations, final RRMS_CG = " << RRMS_CG << endl;
    }

    TOTAL_CG_ITER += ITER_CG;
}

void SOLVER_PRESSURE()
{
    BUILD_RHS_P();
    APPLYIC_CG();
    CALC_RESIDUAL_CG();
    SOLVER_CG();
}

//corrector

void CORRECTOR()
{
    const double facx = deltaT/(rho*DELX);
    const double facy = deltaT/(rho*DELY);

    const double* RESTRICT p_ = P.data();
    const double* RESTRICT us = US.data();
    const double* RESTRICT vs = VS.data();
    double* RESTRICT u = U.data();
    double* RESTRICT v = V.data();

    for(int j=1; j<=NY; j++)
    {
        for(int i=1; i<=NX-1; i++)
        {
            const size_t p = IDU(j,i);

            u[p] = us[p] - facx*(p_[IDP(j,i+1)] - p_[IDP(j,i)]);
        }
    }

    for(int j=1; j<=NY-1; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            const size_t p = IDV(j,i);

            v[p] = vs[p] - facy*(p_[IDP(j+1,i)] - p_[IDP(j,i)]);
        }
    }
}

double UPDATE_TRANSIENT()
{
    const  double invdt = 1.0/deltaT;
    
    const double* RESTRICT u = U.data();
    const double* RESTRICT v = V.data();
    const double* RESTRICT uo = U_old.data();
    const double* RESTRICT vo = V_old.data();

    double* RESTRICT rn = RESN.data();

    for(int j=1; j<=NY; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            const double uc = 0.5*(u[IDU(j,i-1)] + u[IDU(j,i)]);
            const double vc = 0.5*(v[IDV(j-1,i)] + v[IDV(j,i)]);
            const double uc_old = 0.5*(uo[IDU(j,i-1)] + uo[IDU(j,i)]);
            const double vc_old = 0.5*(vo[IDV(j-1,i)] + vo[IDV(j,i)]);

            const double du = (uc - uc_old)*invdt;
            const double dv = (vc - vc_old)*invdt;

            rn[IDP(j,i)] = sqrt(du*du + dv*dv);
        }
    }

    return CALC_NORM(RESN);
}

void CHECK_CONTINUITY()
{
    CALC_DIVERGENCE(U, V);

    double dmax = 0.0;

    for(int j=1; j<=NY; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            dmax = max(dmax, fabs(DIV[IDP(j,i)]));
        }
    }

    if(dmax > WORST_DIV) WORST_DIV = dmax;
       
    if(dmax != dmax)
    {
        cout << "ERROR   : divergence is NaN at step " << TIMESTEP
             << " -- the run has blown up. Reduce CFL_TARGET or VN_SAFETY."
             << endl;
        exit(1);
    }
}

void CHECK_MASS_BALANCE()
{
    double Qw = 0.0, Qe = 0.0, Qs = 0.0, Qn = 0.0;
    
    for(int j=1; j<=NY; j++)
    {
        Qw += U[IDU(j,0)]*AFX;
        Qe -= U[IDU(j,NX)]*AFX;
    }

    for(int i=1; i<=NX; i++)
    {
        Qs += V[IDV(0,i)]*AFY;
        Qn -= V[IDV(NY,i)]*AFY;
    }

    CALC_DIVERGENCE(U, V);

    double dmax = 0.0;
    double dvol = 0.0;

    for(int j=1; j<=NY; j++)
    {
        for(int i=1; i<=NX; i++)
        {
            const double d = DIV[IDP(j,i)];
            dmax = max(dmax, fabs(d));
            dvol += d*VOL;
        }
    }

    const double net = Qw + Qe + Qs + Qn;
    const double scale = fabs(Qw) + fabs(Qe) + fabs(Qs) + fabs(Qn);

    cout << "BALANCE : Qw = " << scientific << setprecision(4) << Qw
         << "   Qe = " << Qe << "   Qs = " << Qs << "   Qn = " << Qn << endl;
    cout << "          net = " << net
         << "   normalised = " << (scale > 0.0 ? net/scale : 0.0)
         << "   (-> 0 for an impermeable cavity)" << endl;
    cout << "          sum(div*VOL) = " << dvol
         << "   (equals the net wall flow by the divergence theorem)" << endl;
    cout << "CONTINUITY: max|div(U)| final = " << dmax
         << "   worst over the run = " << WORST_DIV
         << "   (-> CG tolerance, not discretisation error)" << endl;
    cout << "          RMS ||div|| = " << CALC_NORM(DIV) << endl;
}

void WRITE_FILE_TRANSIENT()
{
    char fname[512];
    snprintf(fname, sizeof(fname), OUTPUT_DIR "/ldc_%05d.dat", TIMESTEP);

    ofstream out(fname);
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << fname << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    out << "TITLE = \"2D Incompressible Navier-Stokes (staggered MAC, Chorin projection)\"" << endl;
    out << "VARIABLES = \"X\", \"Y\", \"U\", \"V\", \"P\", \"VMAG\"" << endl;
    out << "ZONE T=\"t=" << simTime << "\""
        << ", I=" << NX+1 << ", J=" << NY+1
        << ", DATAPACKING=BLOCK, VARLOCATION=([3,4,5,6]=CELLCENTERED)"
        << ", STRANDID=1, SOLUTIONTIME=" << simTime << endl;

    for(int j=0;j<=NY;j++)
        for(int i=0;i<=NX;i++) out << (double)i*DELX << endl;

    for(int j=0;j<=NY;j++)
        for(int i=0;i<=NX;i++) out << (double)j*DELY << endl;

    for(int j=1;j<=NY;j++)
        for(int i=1;i<=NX;i++)
            out << 0.5*(U[IDU(j,i-1)] + U[IDU(j,i)]) << endl;

    for(int j=1;j<=NY;j++)
        for(int i=1;i<=NX;i++)
            out << 0.5*(V[IDV(j-1,i)] + V[IDV(j,i)]) << endl;

    for(int j=1;j<=NY;j++)
        for(int i=1;i<=NX;i++)
            out << P[IDP(j,i)] << endl;

    for(int j=1;j<=NY;j++)
        for(int i=1;i<=NX;i++)
        {
            const double uc = 0.5*(U[IDU(j,i-1)] + U[IDU(j,i)]);
            const double vc = 0.5*(V[IDV(j-1,i)] + V[IDV(j,i)]);
            out << sqrt(uc*uc + vc*vc) << endl;
        }

    out.close();
}

void WRITE_FILE_TRANSIENT_VTK()
{
    char fname[512];
    snprintf(fname, sizeof(fname), OUTPUT_DIR "/ldc_%05d.vtk", TIMESTEP);

    ofstream out(fname);
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << fname << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    out << "# vtk DataFile Version 3.0" << endl;
    out << "2D Incompressible Navier-Stokes (staggered MAC, Chorin projection), t = "
        << simTime << endl;
    out << "ASCII" << endl;
    out << "DATASET RECTILINEAR_GRID" << endl;
    out << "DIMENSIONS " << NX+1 << " " << NY+1 << " " << 1 << endl;

    out << "X_COORDINATES " << NX+1 << " double" << endl;
    for(int i=0;i<=NX;i++) out << (double)i*DELX << endl;

    out << "Y_COORDINATES " << NY+1 << " double" << endl;
    for(int j=0;j<=NY;j++) out << (double)j*DELY << endl;

    out << "Z_COORDINATES 1 double" << endl;
    out << 0.0 << endl;

    out << "CELL_DATA " << NX*NY << endl;

    out << "SCALARS Pressure double 1" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j=1;j<=NY;j++)
        for(int i=1;i<=NX;i++) out << P[IDP(j,i)] << endl;

    out << "SCALARS VMAG double 1" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j=1;j<=NY;j++)
        for(int i=1;i<=NX;i++)
        {
            const double uc = 0.5*(U[IDU(j,i-1)] + U[IDU(j,i)]);
            const double vc = 0.5*(V[IDV(j-1,i)] + V[IDV(j,i)]);
            out << sqrt(uc*uc + vc*vc) << endl;
        }

    out << "VECTORS Velocity double" << endl;
    for(int j=1;j<=NY;j++)
        for(int i=1;i<=NX;i++)
        {
            const double uc = 0.5*(U[IDU(j,i-1)] + U[IDU(j,i)]);
            const double vc = 0.5*(V[IDV(j-1,i)] + V[IDV(j,i)]);
            out << uc << " " << vc << " " << 0.0 << endl;
        }

    out.close();
}
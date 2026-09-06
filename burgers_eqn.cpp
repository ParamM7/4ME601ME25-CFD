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

int NJ = 252;
int NI = 252;

inline size_t ID(int j, int i) { return (size_t)j*NI + i;}
inline size_t IDX(int c, int j, int i) { return ((size_t)c*NJ + j)*NI + i;}

#define SCH_FOU 0
#define SCH_SOU 1
#define SCH_QUICK 2

int SCHEME = SCH_FOU;

const char* SCHEME_NAME()
{
    if (SCHEME == SCH_SOU) return "SOU";
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

void SET_GEOMETRY();

void WRITE_FILE();

void APPLYIC();

void APPLYBC_TEMP();

void CALC_COEFF();
void SOLVER_GS();
void UPDATE();

void APPLYIC_CG();
void APPLYBC_TEMP_CG();
void CALC_RESIDUAL_CG();
void SOLVER_CG();
void WRITE_FILE_CG();
void WRITE_FILE_CG_VTK();

Field XC, YC;
Field U, V;
Field U_old, V_old;
Field FX, FY;
Field UA, VA;
Field UD, VD;
Field CW, CE, CS, CN;
Field CWW, CEE, CSS, CNN;
Field CP;
Field DW, DE, DS, DN;
Field DP;
Field ZEROF;
Field SU;
Field RES;

void ALLOCATE_FIELDS();

double DELX, DELY;
double AFX, AFY;
double VOL;

int k, l, ITER;
int NCELLI;
int NCELLJ;

double LX, LY;
double CCSS;
double RMSRESIDUE;

#define RTOL_GS 1.0e-8
#define MAXITER_GS 10000

long TOTAL_GS_ITER = 0;
double WORST_DIAG = 1.0e30;

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

    for(jj=1;jj<NCELLJ;jj++)
    {
        for(ii=1;ii<NCELLI;ii++)
        {
            const double v = ARR[ID(jj,ii)];
            sumsq = sumsq + v*v;
        }
    }

    return sqrt(sumsq/((double)(NI-2)*(double)(NJ-2)));
}

double CALC_NORM_L1(const Field& ARR)
{
    int ii, jj;
    
    double sumabs = 0.0;

    for(jj=1;jj<NCELLJ;jj++)
    {
        for(ii=1;ii<NCELLI;ii++)
        {
            sumabs = sumabs + fabs(ARR[ID(jj,ii)]);
        }
    }

    return sumabs/((double)(NI-2)*(double)(NJ-2));
}

double simTime;
int TIMESTEP;

#define rho 1.0
#define mu 1.0
#define RE 1
#define U_REF 1.0

double nu;

double deltaT;

#define CFL_TARGET 0.40
#define DT_FLOOR 5.0e-8
#define TMAX 1.0
#define MAXSTEP 200000
double WRITE_INTERVAL = 1e-2; //time interval between writes
double nextWriteTime = WRITE_INTERVAL;
#define DT_ACCURACY 1.0e-3
#define STEADY_TOL  1.0e-9
double io_s = 0.0;

void MAKE_OUTPUT_DIR();
#define OUTPUT_DIR "results"

#define ADV_OPERATOR CW, CE, CS, CN, CWW, CEE, CSS, CNN, CP
#define DIFF_OPERATOR DW, DE, DS, DN, ZEROF, ZEROF, ZEROF, ZEROF, DP

#define AU_WEST 0.0
#define BU_WEST 1.0
#define CU_WEST 0.0

#define AU_SOUTH 0.0
#define BU_SOUTH 1.0
#define CU_SOUTH 0.0

#define AU_EAST 0.0
#define BU_EAST 1.0
#define CU_EAST 0.0

#define AU_NORTH 0.0
#define BU_NORTH 1.0
#define CU_NORTH U_REF

#define AV_WEST   0.0
#define BV_WEST   1.0
#define CV_WEST   0.0

#define AV_SOUTH  0.0
#define BV_SOUTH  1.0
#define CV_SOUTH  0.0

#define AV_EAST   0.0
#define BV_EAST   1.0
#define CV_EAST   0.0

#define AV_NORTH  0.0
#define BV_NORTH  1.0
#define CV_NORTH  0.0

double BC_A_W[2], BC_B_W[2], BC_C_W[2];
double BC_A_S[2], BC_B_S[2], BC_C_S[2];
double BC_A_E[2], BC_B_E[2], BC_C_E[2];
double BC_A_N[2], BC_B_N[2], BC_C_N[2];

void VALIDATE_BC();

void SET_BC_COEFF()
{
    BC_A_W[0] = AU_WEST; BC_B_W[0] = BU_WEST; BC_C_W[0] = CU_WEST;
    BC_A_S[0] = AU_SOUTH; BC_B_S[0] = BU_SOUTH; BC_C_S[0] = CU_SOUTH;
    BC_A_E[0] = AU_EAST; BC_B_E[0] = BU_EAST; BC_C_E[0] = CU_EAST;
    BC_A_N[0] = AU_NORTH; BC_B_N[0] = BU_NORTH; BC_C_N[0] = CU_NORTH;

    BC_A_W[1] = AV_WEST; BC_B_W[1] = BV_WEST; BC_C_W[1] = CV_WEST;
    BC_A_S[1] = AV_SOUTH; BC_B_S[1] = BV_SOUTH; BC_C_S[1] = CV_SOUTH;
    BC_A_E[1] = AV_EAST; BC_B_E[1] = BV_EAST; BC_C_E[1] = CV_EAST;
    BC_A_N[1] = AV_NORTH; BC_B_N[1] = BV_NORTH; BC_C_N[1] = CV_NORTH;

    VALIDATE_BC();
}

void VALIDATE_BC()
{
    // Must run AFTER SET_GEOMETRY: the denominators use the real wall spacings.
    const double dWb = XC[1]    - XC[0];
    const double dEb = XC[NI-1] - XC[NI-2];
    const double dSb = YC[1]    - YC[0];
    const double dNb = YC[NJ-1] - YC[NJ-2];

    const char* wall[4] = {"WEST","SOUTH","EAST","NORTH"};
    const char* comp[2] = {"u","v"};

    for(int c=0; c<2; c++)
    {
        const double den[4] = { BC_A_W[c] + BC_B_W[c]*dWb,
                                BC_A_S[c] + BC_B_S[c]*dSb,
                                BC_A_E[c] + BC_B_E[c]*dEb,
                                BC_A_N[c] + BC_B_N[c]*dNb };

        for(int w=0; w<4; w++)
        {
            if(fabs(den[w]) < 1.0e-30)
            {
                cout << "ERROR: " << wall[w] << " boundary condition on " << comp[c]
                     << " has a + b*d = 0 -- the wall value is undefined."
                     << " Set a or b nonzero." << endl;
                exit(1);
            }
        }
    }

    if(fabs(BC_B_W[0]) < 1.0e-30 || fabs(BC_B_E[0]) < 1.0e-30 ||
       fabs(BC_B_S[1]) < 1.0e-30 || fabs(BC_B_N[1]) < 1.0e-30)
    {
        cout << "NOTE    : at least one wall is zero-gradient in its NORMAL"
             << " component -- the domain is permeable there." << endl;
    }
}

void SET_GEOMETRY();
void MAKE_OUTPUT_DIR();
void APPLYIC();
void APPLYBC(Field& PHI, int comp);
void FILL_CORNERS(Field& PHI);
void SET_DELTAT();
void CALC_FLUXES();
void CALC_COEFF_CONV();
void CALC_COEFF_DIFF();
void BUILD_RHS(const Field& PHI_OLD);
double CALC_RESIDUAL(const Field& AWx,  const Field& AEx,  const Field& ASx,
                     const Field& ANx,  const Field& AWWx, const Field& AEEx,
                     const Field& ASSx, const Field& ANNx, const Field& APx,
                     const Field& PHI);
void SWEEP_GS(const Field& AWx,  const Field& AEx,  const Field& ASx,
              const Field& ANx,  const Field& AWWx, const Field& AEEx,
              const Field& ASSx, const Field& ANNx, const Field& APx,
              Field& PHI);
void SOLVE_GS(const Field& AWx,  const Field& AEx,  const Field& ASx,
              const Field& ANx,  const Field& AWWx, const Field& AEEx,
              const Field& ASSx, const Field& ANNx, const Field& APx,
              const Field& PHI_OLD, Field& PHI, int comp, const char* tag);
void COMBINE();
double UPDATE_TRANSIENT();
void CHECK_DIAG_DOMINANCE();
void WRITE_FILE_TRANSIENT();
void WRITE_FILE_TRANSIENT_VTK();

int main(int argc, char* argv[])
{
    const auto t_wall_start = std::chrono::steady_clock::now();

    CCSS = 1.0e-8;
    LX = 1.0;
    LY = 1.0;

    nu = U_REF*LX/RE;

    if(argc > 1)
    {
        SCHEME = PARSE_SCHEME(argv[1]);
        if(SCHEME < 0)
        {
            cout << "usage: " <<  argv[0] << " [fou|sou|quick] [NI NJ]" << endl;
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

    if (argc > 3)
    {
        NI = atoi(argv[2]);
        NJ = atoi(argv[3]);

        if (NI < 5 || NJ < 5)
        {
            cout << "ERROR: NI and NJ must be >= 5" << endl;
            return -1;
        }
    }

    NCELLI = NI-1;
    NCELLJ = NJ-1;

    cout << "GRID    : NI = " << NI << "   NJ = " << NJ
         << "   -> " << NI-2 << " x " << NJ-2 << " cells" << endl;
    cout << "SCHEME  : " << SCHEME_NAME() << endl;
    cout << "SPLIT   : additive  ->  u^{n+1} = uA + uD - u^n" << endl;
    cout << "PHYSICS : Re = " << RE << "   nu = " << scientific << setprecision(4)
         << nu << endl;

    ALLOCATE_FIELDS();
    MAKE_OUTPUT_DIR();
    SET_GEOMETRY();
    SET_BC_COEFF();

    APPLYIC();
    APPLYBC(U, 0);
    APPLYBC(V, 1);

    CALC_COEFF_DIFF();

    TIMESTEP = 0;
    simTime = 0.0;

    WRITE_FILE_TRANSIENT();
    WRITE_FILE_TRANSIENT_VTK();

    while(simTime < TMAX && TIMESTEP < MAXSTEP)
    {
        std::swap(U, U_old);
        std::swap(V, V_old);

        SET_DELTAT();
        CALC_COEFF_DIFF();

        CALC_FLUXES();
        CALC_COEFF_CONV();

        CHECK_DIAG_DOMINANCE();

        SOLVE_GS(ADV_OPERATOR, U_old, UA, 0, "ADV");
        SOLVE_GS(ADV_OPERATOR, V_old, VA, 1, "ADV");
        SOLVE_GS(DIFF_OPERATOR, U_old, UD, 0, "DIFF");
        SOLVE_GS(DIFF_OPERATOR, V_old, VD, 1, "DIFF");

        COMBINE();

        APPLYBC(U, 0);
        APPLYBC(V, 1);

        const double dUdt = UPDATE_TRANSIENT();

        simTime += deltaT;
        TIMESTEP++;

        if(TIMESTEP % 20 == 0)
        {
            cout << "STEP = " << setw(6) << TIMESTEP
                 << "   t = "  << fixed << setprecision(6) << simTime
                 << "   dt = " << scientific << setprecision(3) << deltaT
                 << "   ||dU/dt|| = " << setprecision(4) << dUdt
                 << "   GS/step = " << TOTAL_GS_ITER/(long)TIMESTEP
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

    const auto t_wall_end = std::chrono::steady_clock::now();
    const double wall_time = std::chrono::duration<double>(t_wall_end - t_wall_start).count();

    cout << "DIAGDOM : worst margin over the run = " << scientific << setprecision(3)
         << WORST_DIAG << endl;

    cout << "Total GS iterations = " << TOTAL_GS_ITER
         << "   (average " << (double)TOTAL_GS_ITER/(double)(TIMESTEP>0?TIMESTEP:1)
         << " per timestep, 4 sub-solves per step)" << endl;

    cout << "WALLTIME: total = " << fixed << setprecision(3) << wall_time << " s"
         << "   I/O = " << io_s << " s"
         << "   compute = " << wall_time - io_s << " s" << endl;

    return 0;
}

void ALLOCATE_FIELDS()
{
    const size_t N = (size_t)NI*NJ;

    XC.assign((size_t)NI, 0.0);
    YC.assign((size_t)NJ, 0.0);

    U.assign(N, 0.0); V.assign(N, 0.0);
    U_old.assign(N, 0.0); V_old.assign(N, 0.0);
    UA.assign(N, 0.0); VA.assign(N, 0.0);
    UD.assign(N, 0.0); VD.assign(N, 0.0);

    FX.assign(N, 0.0); FY.assign(N, 0.0);

    CW.assign(N, 0.0); CE.assign(N, 0.0);
    CS.assign(N, 0.0); CN.assign(N, 0.0);
    CWW.assign(N, 0.0); CEE.assign(N, 0.0);
    CSS.assign(N, 0.0); CNN.assign(N, 0.0);
    CP.assign(N, 0.0);

    DW.assign(N, 0.0); DE.assign(N, 0.0);
    DS.assign(N, 0.0); DN.assign(N, 0.0);
    DP.assign(N, 0.0);

    ZEROF.assign(N, 0.0);

    SU.assign(N, 0.0);
    RES.assign(N, 0.0);
}

void SET_GEOMETRY()
{
    const int NCX = NI-2;
    const int NCY = NJ-2;

    DELX = LX/(double)NCX;
    DELY = LY/(double)NCY;

    AFX = DELY;
    AFY = DELX;
    VOL = DELX*DELY;

    for(int i = 0; i < NI; i++)
    {
        if(i==0) XC[i] = 0.0;
        else if(i == NI - 1) XC[i] = LX;
        else XC[i] = DELX*double(i-1) + 0.5*DELX;
    }

    for(int j = 0; j < NJ; j++)
    {
        if(j==0) YC[j] = 0.0;
        else if(j == NJ - 1) YC[j] = LY;
        else YC[j] = DELY*double(j-1) + 0.5*DELY;
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
        V[p] = 0.0;
        U_old[p] = 0.0;
        V_old[p] = 0.0;
    }
}

void APPLYBC(Field& PHI, int comp)
{
    const double dWb = XC[1] - XC[0];
    const double dEb = XC[NI-1] - XC[NI-2];
    const double dSb = YC[1] - YC[0];
    const double dNb = YC[NJ-1] - YC[NJ-2];

    const double a1 = BC_A_W[comp]; const double b1 = BC_B_W[comp]; const double c1 = BC_C_W[comp];
    const double a2 = BC_A_S[comp]; const double b2 = BC_B_S[comp]; const double c2 = BC_C_S[comp];
    const double a3 = BC_A_E[comp]; const double b3 = BC_B_E[comp]; const double c3 = BC_C_E[comp];
    const double a4 = BC_A_N[comp]; const double b4 = BC_B_N[comp]; const double c4 = BC_C_N[comp];

    //west
    for(int j = 0; j<NJ; j++)
    {
        PHI[ID(j,0)] = (a1*PHI[ID(j,1)] + c1*dWb)/(a1 + b1*dWb);
    }

    //south
    for(int i = 0; i<NCELLI; i++)
    {
        PHI[ID(0,i)] = (a2*PHI[ID(1,i)] + c2*dSb)/(a2 + b2*dSb);
    }

    //east
    for(int j = 0; j<NJ; j++)
    {
        PHI[ID(j,NCELLI)] = (a3*PHI[ID(j,NCELLI-1)] + c3*dEb)/(a3 + b3*dEb);
    }

    //north
    for(int i = 0; i<NCELLI; i++)
    {
        PHI[ID(NCELLJ,i)] = (a4*PHI[ID(NCELLJ-1,i)] + c4*dNb)/(a4 + b4*dNb);
    }

    FILL_CORNERS(PHI);
}

void FILL_CORNERS(Field& PHI)
{
    PHI[ID(0,0)] = 0.5*(PHI[ID(0,1)] + PHI[ID(1,0)]);
    PHI[ID(0,NCELLI)] = 0.5*(PHI[ID(0,NCELLI-1)] + PHI[ID(1,NCELLI)]);
    PHI[ID(NCELLJ,0)] = 0.5*(PHI[ID(NCELLJ-1,0)] + PHI[ID(NCELLJ,1)]);
    PHI[ID(NCELLJ,NCELLI)] = 0.5*(PHI[ID(NCELLJ-1,NCELLI)] + PHI[ID(NCELLJ,NCELLI-1)]);
}

void SET_DELTAT()
{
    double umax = 0.0;

    for(int j=0; j<NI; j++)
    {
        for(int i=0; i<NI; i++)
        {
            const size_t p = ID(j,i);
            umax = max(umax, fabs(U_old[p]));
            umax = max(umax, fabs(V_old[p]));

        }
    }

    double dt = DT_ACCURACY;

    if(umax > 1.0e-12)
    {
        dt = min(dt, CFL_TARGET*min(DELX,DELY)/umax);
    }

    deltaT = max(dt, DT_FLOOR);

    if(simTime + deltaT > TMAX) deltaT = TMAX - simTime;
}

void CALC_FLUXES()
{
    for(int j=0; j<NCELLJ; j++)
    {
        for(int i=0; i<NCELLI; i++)
        {
            const size_t p = ID(j,i);
            double uf;

            if(i==0) uf = U_old[ID(j,0)];
            else if(i==NI-2) uf = U_old[ID(j,NI-1)];
            else uf = 0.5*(U_old[p] + U_old[p+1]);
            
            FX[p] = uf*AFX;
        }
    }

    for(int j=0; j<NCELLJ; j++)
    {
        for(int i=0; i<NCELLI; i++)
        {
            const size_t p = ID(j,i);
            double vf;

            if(i==0) vf = V_old[ID(0,i)];
            else if(i==NI-2) vf = V_old[ID(NJ-1,i)];
            else vf = 0.5*(V_old[p] + V_old[p+NI]);
            
            FY[p] = vf*AFY;
        }
    }
}

inline void FACE_WEIGHTS(double& wU, double& wD, double& wUU)
{
    if(SCHEME == SCH_SOU) { wU = 1.500; wD = 0.000; wUU = -0.500; return; }
    if(SCHEME == SCH_QUICK) { wU = 0.750; wD = 0.375; wUU = -0.125; return; }

    wU = 1.000; wD = 0.000; wUU = 0.000;
}

void CALC_COEFF_CONV()
{
    const double tr = VOL/deltaT;

    double wU, wD, wUU;
    FACE_WEIGHTS(wU, wD, wUU);

    for(int j=0; j<NCELLJ; j++)
    {
        for(int i=0; i<NCELLI; i++)
        {
            const size_t p = ID(j,i);

            const double Fe = FX[p];
            const double Fw = FX[p-1];
            const double Fn = FY[p];
            const double Fs = FY[p-NI];

            double aP = tr;
            double aW = 0.0, aE = 0.0, aS = 0.0, aN = 0.0;
            double aWW = 0.0, aEE = 0.0, aSS = 0.0, aNN = 0.0;

            if(i == NCELLI-1)
            {
                aE -= Fe;
            }
            else if(Fe >= 0.0) // U = P, D = E, UU = W
            {
                aP += Fe*wU;
                aE -= Fe*wD;
                aW -= Fe*wUU;
            }
            else // U = E, D = P, UU = EE
            {
                aE -= Fe*wU;
                aP += Fe*wD;
                aEE -= Fe*wUU;
            }

            if(i == 1)
            {
                aW += Fw;
            }
            else if(Fw >= 0.0)  // U = W, D = P, UU = WW
            {
                aW += Fw*wU;
                aP -= Fw*wD;
                aWW += Fw*wUU;
            }
            else // U = P, D = W, UU = E
            {
                aP -= Fw*wU;
                aW += Fw*wD;
                aE += Fw*wUU;
            }

            if(j == NCELLJ-1)
            {
                aN -= Fn;
            }
            else if(Fn >= 0.0)
            {
                aP += Fn*wU;
                aN -= Fn*wD;
                aNN -= Fn*wUU;
            }
            else
            {
                aN -= Fn*wU;
                aP += Fn*wD;
                aNN += Fn*wUU;
            }

            if(j==1)
            {
                aS += Fs;
            }
            else if(Fs >= 0.0)
            {
                aS += Fs*wU;
                aP -= Fs*wD;
                aSS += Fs*wUU;
            }
            else
            {
                aP -= Fs*wU;
                aS += Fs*wD;
                aN += Fs*wUU;
            }

            CW[p] = aW; CE[p] = aE; CS[p] = aS; CN[p] = aN;
            CWW[p] = aWW; CEE[p] = aEE; CSS[p] = aSS; CNN[p] = aNN;
            CP[p] = aP;
        }
    }
}

void CALC_COEFF_DIFF()
{
    const double tr = VOL/deltaT;

    for(int j=0; j<NCELLJ;j++)
    {
        const double dS = YC[j] - YC[j-1];
        const double dN = YC[j+1] - YC[j];

        for(int i=0; i<NCELLI; i++)
        {
            const size_t p = ID(j,i);

            const double dWw = XC[i] - XC[i-1];
            const double dEe = XC[i+1] - XC[i];

            DW[p] = nu*AFX/dWw;
            DE[p] = nu*AFX/dEe;
            DS[p] = nu*AFY/dS;
            DN[p] = nu*AFY/dN;

            DP[p] = tr + DW[p] + DE[p] + DS[p] + DN[p];
        }
    }
}

void CHECK_DIAG_DOMINANCE()
{
    // With SOU/QUICK assembled directly the off-diagonals change sign, so the
    // meaningful test is  aP - sum|a_nb| >= 0, the sufficient condition for
    // Gauss-Seidel convergence. Under FOU it collapses back to the old
    // aP - sum(a_nb) test, because every a_nb is then non-negative anyway.
    double worst = 1.0e30;

    for(int j=1;j<NCELLJ;j++)
    {
        for(int i=1;i<NCELLI;i++)
        {
            const size_t p = ID(j,i);

            const double sumabs = fabs(CW[p])  + fabs(CE[p])
                                + fabs(CS[p])  + fabs(CN[p])
                                + fabs(CWW[p]) + fabs(CEE[p])
                                + fabs(CSS[p]) + fabs(CNN[p]);

            worst = min(worst, CP[p] - sumabs);
        }
    }

    // Dominance depends on dt AND on the instantaneous flux field, so this is
    // checked every step and the running minimum is reported at the end.
    if(worst < WORST_DIAG) WORST_DIAG = worst;

    if(TIMESTEP == 0 || (worst < 0.0 && TIMESTEP % 100 == 0))
    {
        cout << "DIAGDOM : min( CP - sum|Cnb| ) = " << scientific << setprecision(3) << worst
             << (worst >= 0.0 ? "   -> GS convergence guaranteed"
                              : "   -> WARNING: not diagonally dominant, reduce dt")
             << endl;
    }
}

void BUILD_RHS(const Field& PHI_OLD)
{
    const double tr = VOL/deltaT;

    for(int j=1;j<NCELLJ;j++)
    {
        for(int i=1; i<NCELLI; i++)
        {
            const size_t p = ID(j,i);
            SU[p] = tr*PHI_OLD[p];
        }
    }
}

#define PWW ((i>1) ? p-2 :p)
#define PEE ((i<NCELLI-1) ? p+2 : p)
#define PSS ((j>1) ? p-2*ni :p)
#define PNN ((j<NCELLJ-1) ? p+2*ni : p)

double CALC_RESIDUAL(const Field& AWx, const Field& AEx, const Field& ASx, const Field& ANx, const Field& AWWx, const Field& AEEx , const Field& ASSx, const Field& ANNx, const Field& APx, const Field& PHI)
{
    const int ni = NI;

    for(int j =1; j < NCELLJ; j++)
    {
        for(int i=1; i < NCELLI; i++)
        {
            const size_t p = ID(j,i);

            RES[p] = SU[p]
                     + AWx[p]*PHI[p-1] + AEx[p]*PHI[p+1]
                     + ASx[p]*PHI[p-ni] + ANx[p]*PHI[p+ni]
                     + AWWx[p]*PHI[PWW] + AEEx[p]*PHI[PEE]
                     + ASSx[p]*PHI[PSS] + ANNx[p]*PHI[PNN]
                        - APx[p]*PHI[p];
        }
    }

    return CALC_NORM(RES);
}

void SWEEP_GS(const Field& AWx, const Field& AEx, const Field& ASx, const Field& ANx, const Field& AWWx, const Field& AEEx, const Field& ASSx, const Field& ANNx, const Field& APx, Field& PHI)
{
    const int ni = NI;

    const double* RESTRICT aw = AWx.data();
    const double* RESTRICT ae = AEx.data();
    const double* RESTRICT as = ASx.data();
    const double* RESTRICT an = ANx.data();
    const double* RESTRICT aww = AWWx.data();
    const double* RESTRICT aee = AEEx.data();
    const double* RESTRICT ass = ASSx.data();
    const double* RESTRICT ann = ANNx.data();
    const double* RESTRICT ap = APx.data();
    const double* RESTRICT su = SU.data();
    double* RESTRICT phi = PHI.data();

    for(int j=1; j<NCELLJ;j++)
    {
        for(int i=1; i<NCELLI;i++)
        {
            const size_t p = ID(j,i);
            phi[p] = ( aw[p]*phi[p-1] + ae[p]*phi[p+1]
                     + as[p]*phi[p-ni] + an[p]*phi[p+ni]
                     + aww[p]*phi[PWW] + aee[p]*phi[PEE]
                     + ass[p]*phi[PSS] + ann[p]*phi[PNN]
                     + su[p] ) / ap[p];
        }
    }
}

void SOLVE_GS(const Field& AWx,  const Field& AEx,  const Field& ASx, const Field& ANx,  const Field& AWWx, const Field& AEEx, const Field& ASSx, const Field& ANNx, const Field& APx, const Field& PHI_OLD, Field& PHI, int comp, const char* tag)
{
    PHI = PHI_OLD;
    APPLYBC(PHI, comp);

    BUILD_RHS(PHI_OLD);

    double res0 = -1.0;
    int it = 0;

    for(it=1; it<=MAXITER_GS; it++)
    {
        SWEEP_GS(AWx, AEx, ASx, ANx, AWWx, AEEx, ASSx, ANNx, APx, PHI);
        APPLYBC(PHI, comp);

        RMSRESIDUE = CALC_RESIDUAL(AWx, AEx, ASx, ANx, AWWx, AEEx, ASSx, ANNx, APx, PHI);

        if(res0 < 0.0) res0 = RMSRESIDUE;

        if(VERBOSE && it % 20 == 0)
        cout << "   " << tag << " it = " << it
                 << "   res = " << RMSRESIDUE << endl;

        if(RMSRESIDUE < CCSS || RMSRESIDUE < RTOL_GS*res0) break;
    }

    if(it > MAXITER_GS)
        cout << "WARNING: " << tag << " GS hit MAXITER_GS at step " << TIMESTEP
             << "   res = " << RMSRESIDUE << endl;

    TOTAL_GS_ITER += it;
}

void COMBINE()
{
    for(int j=1; j<NCELLJ;j++)
    {
        for(int i=1; i<NCELLI;i++)
        {
            const size_t p = ID(j,i);

            U[p] = UA[p] + UD[p] - U_old[p];
            V[p] = VA[p] + VD[p] - V_old[p];
        }
    }
}

double UPDATE_TRANSIENT()
{
    const double invdt = 1.0/deltaT;

    for(int j=1; j<NCELLJ; j++)
    {
        for(int i=1;i<NCELLI;i++)
        {
            const size_t p = ID(j,i);
            const double du = (U[p] - U_old[p])*invdt;
            const double dv = (V[p] - V_old[p])*invdt;
            RES[p] = sqrt(du*du + dv*dv);
        }
    }

    return CALC_NORM(RES);
}

void WRITE_FILE_TRANSIENT()
{
    char fname[512];
    snprintf(fname, sizeof(fname), OUTPUT_DIR "/burgers_%05d.dat", TIMESTEP);

    ofstream out(fname);
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << fname << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    const int NCX = NI-2;
    const int NCY = NJ-2;

    out << "TITLE = \"2D Velocity Burgers (cell-centred FVM, GS)\"" << endl;
    out << "VARIABLES = \"X\", \"Y\", \"U\", \"V\", \"VMAG\"" << endl;
    out << "ZONE T=\"t=" << simTime << "\""
        << ", I=" << NCX+1 << ", J=" << NCY+1
        << ", DATAPACKING=BLOCK, VARLOCATION=([3,4,5]=CELLCENTERED)"
        << ", STRANDID=1, SOLUTIONTIME=" << simTime << endl;

    for(int j=0;j<=NCY;j++)
        for(int i=0;i<=NCX;i++) out << (double)i*DELX << endl;

    for(int j=0;j<=NCY;j++)
        for(int i=0;i<=NCX;i++) out << (double)j*DELY << endl;

    for(int j=1;j<NCELLJ;j++)
        for(int i=1;i<NCELLI;i++) out << U[ID(j,i)] << endl;

    for(int j=1;j<NCELLJ;j++)
        for(int i=1;i<NCELLI;i++) out << V[ID(j,i)] << endl;

    for(int j=1;j<NCELLJ;j++)
        for(int i=1;i<NCELLI;i++)
        {
            const size_t p = ID(j,i);
            out << sqrt(U[p]*U[p] + V[p]*V[p]) << endl;
        }

    out.close();
}

void WRITE_FILE_TRANSIENT_VTK()
{
    char fname[512];
    snprintf(fname, sizeof(fname), OUTPUT_DIR "/burgers_%05d.vtk", TIMESTEP);

    ofstream out(fname);
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << fname << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    const int NCX = NI-2;
    const int NCY = NJ-2;

    out << "# vtk DataFile Version 3.0" << endl;
    out << "2D Velocity Burgers (cell-centred FVM, GS), t = " << simTime << endl;
    out << "ASCII" << endl;
    out << "DATASET RECTILINEAR_GRID" << endl;
    out << "DIMENSIONS " << NCX+1 << " " << NCY+1 << " " << 1 << endl;

    out << "X_COORDINATES " << NCX+1 << " double" << endl;
    for(int i=0;i<=NCX;i++) out << (double)i*DELX << endl;

    out << "Y_COORDINATES " << NCY+1 << " double" << endl;
    for(int j=0;j<=NCY;j++) out << (double)j*DELY << endl;

    out << "Z_COORDINATES 1 double" << endl;
    out << 0.0 << endl;

    out << "CELL_DATA " << NCX*NCY << endl;

    out << "SCALARS VMAG double 1" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for(int j=1;j<NCELLJ;j++)
        for(int i=1;i<NCELLI;i++)
        {
            const size_t p = ID(j,i);
            out << sqrt(U[p]*U[p] + V[p]*V[p]) << endl;
        }

    out << "VECTORS Velocity double" << endl;
    for(int j=1;j<NCELLJ;j++)
        for(int i=1;i<NCELLI;i++)
            out << U[ID(j,i)] << " " << V[ID(j,i)] << " " << 0.0 << endl;

    out.close();
}
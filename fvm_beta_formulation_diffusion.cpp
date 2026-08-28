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

void SET_GEOMETRY();

void CALC_COEFF();

void APPLYIC_CG();
void APPLYBC_TEMP_CG();
void CALC_RESIDUAL_CG();
void SOLVER_CG();
void WRITE_FILE_CG();
void WRITE_FILE_CG_VTK();

void APPLYIC_TRANSIENT();
void FILL_CORNERS();
void REPORT_POSITIVITY();
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
void CHECK_FLUX_BALANCE();
void SET_SOURCE();

double BETA = 1.0;

#define OUTPUT_DIR "results"

#define BETA_ZERO_TOL 1.0e-14

bool VERBOSE = false;

Field XCELL;
Field XC, YC;
Field SP;
Field T, T_old_time;
Field AW, AE, AS, AN, AP;
Field AW1, AE1, AS1, AN1, AP1;
Field TCG, RES, PDIR, AP_CG;
Field QVOL;
Field QSLOPE;

void ALLOCATE_FIELDS();

double DELX, DELY;
double AFX, AFY;
double VOL;

int k, l, ITER;
int NCELLI;
int NCELLJ;

double LX,LY;
double CCSS; //convergence criteria
double RRMS, RMSRESIDUE;
long TOTAL_CG_ITER = 0;

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

//CONJUGATE GRADIENT

#define MAXITER_CG 50000

double ALPHA_CG; //ALPHA_CG = (r.r)/(p.A*p)

double BETA_CG; // (r_new.r_new)/(r_old.r_old)

double RS_OLD, RS_NEW; // r.r

double PAP_CG; //p.(A*p)

double RRMS_CG;

int ITER_CG;

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
#define Q_GEN 10.0;
#define Q_SLOPE 0.0;

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

void MAKE_OUTPUT_DIR();

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

    cout<< "GRID    : NI = " << NI << "   NJ = " << NJ
        << "   -> " << NI-2 << " x " << NJ-2 << " cells" << endl;
    cout<< "          (" << (size_t)(NI-2)*(NJ-2) << " unknowns, "
        << 2*((NI-2)+(NJ-2)) << " boundary face nodes)" << endl;
    
    ALLOCATE_FIELDS();

    MAKE_OUTPUT_DIR();

    SET_GEOMETRY();

    SET_SOURCE();

    SET_DELTAT();

    CALC_FOURIER();
    CHECK_STABILITY();

    APPLYIC_TRANSIENT();
    APPLYBC_TEMP_TRANSIENT();

    CALC_COEFF_TRANSIENT();

    TIMESTEP = 0;
    simTime = 0.0;

    WRITE_FILE_TRANSIENT();

    while(simTime < TMAX && TIMESTEP <MAXSTEP)
    {
        std::swap(T, T_old_time);

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

    WRITE_FILE_TRANSIENT();
    WRITE_FILE_TRANSIENT_VTK();

    CHECK_FLUX_BALANCE();

    if(BETA >= BETA_ZERO_TOL)
    {
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
    
    return (0);
    
}

void ALLOCATE_FIELDS()
{
    const size_t N = (size_t)NI*NJ;

    XCELL.assign(2*N, 0.0);
    XC.assign((size_t)NI, 0.0);
    YC.assign((size_t)NJ, 0.0);
    SP.assign(N, 0.0);
    QVOL.assign(N, 0.0);
    QSLOPE.assign(N, 0.0);

    T.assign(N, 0.0);
    T_old_time.assign(N, 0.0);

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

    TCG.assign(N, 0.0);
    RES.assign(N, 0.0);
    PDIR.assign(N, 0.0);
    AP_CG.assign(N, 0.0);
}

void SET_GEOMETRY()
{   
    const int NCX = NI-2;
    const int NCY = NJ-2;

    DELX = LX/(double)NCX;
    DELY = LY/(double)NCY;
    AFX = DELX;
    AFY = DELY;
    VOL = DELX*DELY;

    for(int i=0; i<NI; i++)
    {
        if(i==0)    XC[i] = 0.0;
        else if(i==NI-1) XC[i] = LX;
        else XC[i] = DELX*(double)(i-1) + 0.5*DELX;
    }

    for(int j=0; j<NJ; j++)
    {
        if(j==0)    YC[j] = 0.0;
        else if(j==NJ-1) YC[j] = LY;
        else YC[j] = DELY*(double)(j-1) + 0.5*DELY;
    }

    for(int j=0; j<NJ; j++){
       for(int i=0;i<NI;i++){
        XCELL[IDX(0,j,i)] = XC[i];
        XCELL[IDX(1,j,i)] = YC[j];
       }
     }
}

void SET_SOURCE()
{
    for(int j=1; j<NCELLJ; j++)
    {
        for(int i=1; i<NCELLI; i++)
        {
            const size_t p = ID(j,i);

            QVOL[p] = Q_GEN;
            QSLOPE[p] = Q_SLOPE;
            
            if(QSLOPE[p] > 0.0)
            {
                cout << "ERROR: QSLOPE must be <= 0 (Patankar). Got "
                     << QSLOPE[p] << " at (" << j << "," << i << ")." << endl;
                exit(1);
            }
        }
    }
}

void CALC_COEFF()
{
    for(int j=1;j<NCELLJ;j++)
    {
        const double dS = YC[j] - YC[j-1];
        const double dN = YC[j+1] - YC[j];

        for(int i=1;i<NCELLI;i++)
        {
            const double dW = XC[i] - XC[i-1];
            const double dE = XC[i+1] - XC[i];
            
            const size_t p = ID(j,i);

            AW[p] = kT*AFX/(dW*VOL);
            AE[p] = kT*AFX/(dE*VOL);
            AS[p] = kT*AFX/(dS*VOL);
            AN[p] = kT*AFX/(dN*VOL);

            AP[p] = AW[p]+AE[p]+AS[p]+AN[p];
        }
    }
}


//CONJUGATE GRADIENT

void APPLYIC_CG()
{
    for(int j=0;j<NJ;j++)
    {
        for(int i=0;i<NI;i++)
        {
            const size_t p = ID(j,i);

            TCG[p] = T_old_time[p];
            RES[p] = 0.0;
            PDIR[p] = 0.0;
            AP_CG[p] = 0.0;
        }
    }
}

void APPLYBC_TEMP_CG()
{
    const double dWb = XC[1] - XC[0];
    const double dSb = YC[1] - YC[0];
    const double dEb = XC[NI-1] - XC[NI-2];
    const double dNb = YC[NJ-1] - YC[NJ-2];

    //using generalized boundary condition c+a(dT/dn)=bT
    double a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;

    //west
    a1 = A_WEST;b1=B_WEST;c1=C_WEST;
    for(int j=0;j<NJ;j++)
    {
       TCG[ID(j,0)] = (a1*TCG[ID(j,1)]+c1*dWb)/(a1+b1*dWb);
    }
    //south
    a2 = A_SOUTH;b2=B_SOUTH;c2=C_SOUTH;
    for(int i=1;i<NCELLI;i++)
    {       
        TCG[ID(0,i)] = (a2*TCG[ID(1,i)]+c2*dSb)/(a2+b2*dSb);
    }
    //east
    a3 = A_EAST;b3=B_EAST;c3=C_EAST;
    for(int j=0;j<NJ;j++)
    {        
        TCG[ID(j,NCELLI)] = (a3*TCG[ID(j,NCELLI-1)]+c3*dEb)/(a3+b3*dEb);
    }
    //north
    a4 = A_NORTH;b4=B_NORTH;c4=C_NORTH;
    for(int i=1;i<NCELLI;i++)
    {        
        TCG[ID(NCELLJ,i)] = (a4*TCG[ID(NCELLJ-1,i)]+c4*dNb)/(a4+b4*dNb);
    }

}

void CALC_RESIDUAL_CG()
{
    const int ni=NI, ncelli = NCELLI, ncellj = NCELLJ;

    const double* RESTRICT ap = AP.data();
    const double* RESTRICT aw = AW.data();
    const double* RESTRICT an = AN.data();
    const double* RESTRICT ae = AE.data();
    const double* RESTRICT as = AS.data();
    const double* RESTRICT tcg = TCG.data();
    const double* RESTRICT sp = SP.data();
    double* RESTRICT res = RES.data();
    double* RESTRICT pdir = PDIR.data();

    double rs = 0.0;

    for(int j=1;j<ncellj;j++)
    {
        for(int i=1;i<ncelli;i++)
        {
            const size_t p = ID(j,i);

            double Ax = ap[p]*tcg[p]
                        - aw[p]*tcg[p-1] - ae[p]*tcg[p+1]
                        - as[p]*tcg[p-NI] - an[p]*tcg[p+NI];

            res[p] = sp[p] - Ax; //r = b-Ax
            pdir[p] = res[p]; //p0 = r0

            rs = rs + res[p]*res[p]; //r.r
        }
    }

    RS_OLD = rs;
}

void SOLVER_CG()
{
    //Solves A x = b by Conjugate Gradient. Knows nothing about time -- the physical
    //timestep loop lives in main(), and T_old_time is FROZEN for this entire routine.
    //
    //  r_0     = b - A x_0
    //  p_0     = r_0                          (both set in CALC_RESIDUAL_CG)
    //  alpha_k = (r_k . r_k)/(p_k . A p_k)
    //  x_{k+1} = x_k + alpha_k p_k
    //  r_{k+1} = r_k - alpha_k A p_k
    //  beta_k  = (r_{k+1} . r_{k+1})/(r_k . r_k)
    //  p_{k+1} = r_{k+1} + beta_k p_k

    //With the warm start (TCG = T^n) the initial residual can already sit below
    //tolerance. Entering the loop would then divide RS_OLD by a vanishing PAP_CG and
    //push inf/NaN straight into TCG.

    const int ni = NI, ncelli = NCELLI, ncellj = NCELLJ;
    const double invN = 1.0/((double)(NI-2)*(double)(NJ-2));

    const double* RESTRICT ap = AP.data();
    const double* RESTRICT aw = AW.data();
    const double* RESTRICT ae = AE.data();
    const double* RESTRICT as = AS.data();
    const double* RESTRICT an = AN.data();
    double* RESTRICT pd   = PDIR.data();
    double* RESTRICT apcg = AP_CG.data();
    double* RESTRICT tcg  = TCG.data();
    double* RESTRICT res  = RES.data();


    RRMS_CG = sqrt(RS_OLD*invN);
    if(RRMS_CG < CCSS)
    {
        ITER_CG = 0;
        return;
    }

    for(ITER_CG=1; ITER_CG<=MAXITER_CG; ITER_CG++)
    {
        // Ap = A*p, with p.Ap accumulated in the same sweep.
        //Matrix-free: A is never stored, only applied as a 5-point stencil. PDIR is
        //zero on the boundary (APPLYIC_CG zeroes the full field and only interior
        //nodes are ever written), so those neighbour terms drop out cleanly -- which
        //is exactly the homogeneous-Dirichlet operator CG needs.
        double pap = 0.0;

        for(int j=1; j<ncellj; j++)
        {
            const size_t row = (size_t)j*ni;

            for(int i=1; i<ncelli; i++)
            {
                const size_t p = row + i;

                apcg[p] = ap[p]*pd[p]
                        - aw[p]*pd[p-1]  - ae[p]*pd[p+1]
                        - as[p]*pd[p-ni] - an[p]*pd[p+ni];

                pap += pd[p]*apcg[p];
            }
        }

        PAP_CG = pap;

        //For an SPD matrix p.Ap > 0 strictly. Zero means either convergence to machine
        //precision or a broken (non-symmetric) operator -- stop either way.
        if(fabs(PAP_CG) < 1.0e-30)
        {
            break;
        }

        ALPHA_CG = RS_OLD/PAP_CG;

        //update solution, x and residual, r simultaneously
        //The recurrence r <- r - alpha*Ap avoids recomputing b - Ax each iteration,
        //saving one full stencil sweep per iteration.
        double rsnew = 0.0;

        for(int j=1; j<ncellj; j++)
        {
            const size_t row = (size_t)j*ni;

            for(int i=1; i<ncelli; i++)
            {
                const size_t p = row + i;

                tcg[p] = tcg[p] + ALPHA_CG*pd[p];
                res[p] = res[p] - ALPHA_CG*apcg[p];

                rsnew += res[p]*res[p];
            }
        }

        RS_NEW = rsnew;

        //rsnew IS the sum CALC_NORM_L2 would recompute -- same terms, same order.
        RRMS_CG = sqrt(RS_NEW*invN);

        if(RRMS_CG < CCSS)
        {
            break;
        }

        //BETA_CG to update PDIR.
        //beta is what keeps successive directions A-conjugate; drop it and this
        //degenerates into steepest descent.
        BETA_CG = RS_NEW/RS_OLD;

        for(int j=1; j<ncellj; j++)
        {
            const size_t row = (size_t)j*ni;

            for(int i=1; i<ncelli; i++)
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
        cout << "WARNING: CG hit MAXITER_CG at timestep " << TIMESTEP
             << "   RRMS_CG = " << RRMS_CG << endl;
    }

    

    TOTAL_CG_ITER = TOTAL_CG_ITER + ITER_CG;
}

void WRITE_FILE_CG()
{
    ofstream out(OUTPUT_DIR "/temperature_field_CG.dat");
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << OUTPUT_DIR "/temperature_field_CG.dat"
             << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    out << "TITLE = \"2D Steady-State Heat Diffusion (CONJUGATE GRADIENT FDM)\"" << endl;
    out << "VARIABLES = \"X\", \"Y\", \"T\"" << endl;
    out << "ZONE T=\"Steady Heat Diffusion\", I=" << NI << ", J=" << NJ
        << ", F=POINT" << endl;

    for (int j = 0; j < NJ; j++)
    {
        for (int i = 0; i < NI; i++)
        {
            out << XCELL[IDX(0,j,i)] << " " << XCELL[IDX(1,j,i)] << " "
                << TCG[ID(j,i)] << endl;
        }
    }

    out.close();

    cout << "Field written to temperature_field_CG.dat (Tecplot ASCII)" << endl;
}

void WRITE_FILE_CG_VTK()
{
    ofstream out(OUTPUT_DIR "/temperature_field_CG.vtk");
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << OUTPUT_DIR "/temperature_field_CG.vtk"
             << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    out << "# vtk DataFile Version 3.0" << endl;
    out << "2D Steady-State Heat Diffusion (Conjugate Gradient FDM)" << endl;
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
            out << TCG[ID(j,i)] << endl;
        }
    }

    out.close();

    cout << "Field written to temperature_field_CG.vtk (ParaView Legacy VTK)" << endl;
}

//define the transient heat conduction problem with explicit time marching scheme

void APPLYIC_TRANSIENT()
{
    for(int j=0;j<NJ;j++)
    {
        for(int i=0;i<NI;i++)
        {
            T[ID(j,i)] = T_INIT;
            T_old_time[ID(j,i)] = T_INIT;
        }
    }
}

void APPLYBC_TEMP_TRANSIENT()
{
    const double dWb = XC[1] - XC[0];
    const double dSb = YC[1] - YC[0];
    const double dEb = XC[NI-1] - XC[NI-2];
    const double dNb = YC[NJ-1] - YC[NJ-2];

    double a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;

    //west
    a1 = A_WEST;b1=B_WEST;c1=C_WEST;
    for(int j=0;j<NJ;j++)
    {
        T[ID(j,0)] = (a1*T[ID(j,1)]+c1*dWb)/(a1+b1*dWb);
        T_old_time[ID(j,0)] = (a1*T_old_time[ID(j,1)]+c1*dWb)/(a1+b1*dWb);
    }
    //south
    a2 = A_SOUTH;b2=B_SOUTH;c2=C_SOUTH;
    for(int i=1;i<NCELLI;i++)
    {
        T[ID(0,i)] = (a2*T[ID(1,i)]+c2*dSb)/(a2+b2*dSb);
        T_old_time[ID(0,i)] = (a2*T_old_time[ID(1,i)]+c2*dSb)/(a2+b2*dSb);
    }
    //east
    a3 = A_EAST;b3=B_EAST;c3=C_EAST;
    for(int j=0;j<NJ;j++)
    {
        T[ID(j,NCELLI)] = (a3*T[ID(j,NCELLI-1)]+c3*dEb)/(a3+b3*dEb);
        T_old_time[ID(j,NCELLI)] = (a3*T_old_time[ID(j,NCELLI-1)]+c3*dEb)/(a3+b3*dEb);
    }
    //north
    a4 = A_NORTH;b4=B_NORTH;c4=C_NORTH;
    for(int i=1;i<NCELLI;i++)
    {
        T[ID(NCELLJ,i)] = (a4*T[ID(NCELLJ-1,i)]+c4*dNb)/(a4+b4*dNb);
        T_old_time[ID(NCELLJ,i)] = (a4*T_old_time[ID(NCELLJ-1,i)]+c4*dNb)/(a4+b4*dNb);
    }

    FILL_CORNERS();
}

void FILL_CORNERS()
{
   T[ID(0,0)] = 0.5*(T[ID(0,1)] + T[ID(1,0)]);
   T[ID(0,NI-1)] = 0.5*(T[ID(0,NI-2)] + T[ID(1,NI-1)]);
   T[ID(NJ-1,0)] = 0.5*(T[ID(NJ-1,1)] + T[ID(NJ-2,0)]);
   T[ID(NJ-1,NI-1)] = 0.5*(T[ID(NJ-1,NI-2)] + T[ID(NJ-2,NI-1)]);

}

void REPORT_POSITIVITY()
{
    if(BETA >=1 -1e-12)
    {
        return;
    }

    const double pos_lim = 1.0/(3.0*(1.0-BETA));
    const double Fo_sum  = Fo_x + Fo_y;

    cout << "POSITIV : corner-cell limit Fo_x+Fo_y <= " << fixed << setprecision(6)
         << pos_lim
         << (Fo_sum > pos_lim ? "   -> VIOLATED (oscillation possible)"
                              : "   -> satisfied") << endl;
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
        REPORT_POSITIVITY();
        return;
    }

    double dt_max = Fo_lim/(alpha*(1.0/(DELX*DELX) + 1.0/(DELY*DELY)));

    cout << "STABIL  : Fo_x+Fo_y = " << fixed << setprecision(6) << Fo_sum
         << "   limit = " << Fo_lim
         << "   dt_max = " << scientific << setprecision(4) << dt_max << endl;

    REPORT_POSITIVITY();

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
    const double fac = alpha*deltaT/VOL;
    double sum_max = 0.0;

    for(int j=1;j<NCELLJ;j++)
    {
        const double dS = YC[j] - YC[j-1];
        const double dN = YC[j+1] - YC[j];

        for(int i=1;i<NCELLI;i++)
        {

            const size_t p = ID(j,i);

            const double dW = XC[i] - XC[i-1];
            const double dE = XC[i+1] - XC[i];

            const double Dw = fac*AFX/dW;
            const double De = fac*AFX/dE;
            const double Ds = fac*AFY/dS;
            const double Dn = fac*AFY/dN;
            const double Ssp = -QSLOPE[p]*deltaT/(rho*Cp);
            const double Dsum = Dw + De + Ds + Dn + Ssp;

            AW[p] = BETA*Dw;
            AE[p] = BETA*De;
            AN[p] = BETA*Dn;
            AS[p] = BETA*Ds;

            AP[p] = 1.0 + BETA*Dsum;

            AW1[p] = (1.0-BETA)*Dw;
            AE1[p] = (1.0-BETA)*De;
            AN1[p] = (1.0-BETA)*Dn;
            AS1[p] = (1.0-BETA)*Ds;

            AP1[p] = 1.0 - (1.0-BETA)*Dsum;

            if(Dsum > sum_max) sum_max = Dsum;
        }
    }
    
    const double ap1_min = 1.0 - (1.0-BETA)*sum_max;

    if(ap1_min < 0.0)
    {
        cout << "WARNING : min AP1 = " << fixed << setprecision(4) << ap1_min
             << " < 0 violates Patankar's positive-coefficient rule at the corner"
             << " cells. The scheme is still stable but the solution may oscillate;"
             << " non-oscillatory behaviour needs 3*(Fo_x+Fo_y) <= 1/(1-BETA)."
             << endl;
    }
}

void BUILD_RHS_TRANSIENT()
{
    const int ni = NI, ncelli = NCELLI, ncellj = NCELLJ;

    const double* RESTRICT ap1 = AP1.data();
    const double* RESTRICT aw1 = AW1.data();
    const double* RESTRICT ae1 = AE1.data();
    const double* RESTRICT as1 = AS1.data();
    const double* RESTRICT an1 = AN1.data();
    const double* RESTRICT told = T_old_time.data();
    const double* RESTRICT qv = QVOL.data();
    double* RESTRICT sp = SP.data();

    const double qfac = deltaT/(rho*Cp);

    for(int j=1; j<ncellj; j++)
    {
        const size_t row = (size_t)j*ni;

        for(int i=1; i<ncelli; i++)
        {
            const size_t p = row + i;

            sp[p] = ap1[p]*told[p]
                  + aw1[p]*told[p-1]  + ae1[p]*told[p+1]
                  + as1[p]*told[p-ni] + an1[p]*told[p+ni] + qfac*qv[p];
        }
    }
}

void SOLVER_TRANSIENT()
{
    BUILD_RHS_TRANSIENT();

    if(BETA < BETA_ZERO_TOL)
    {
        for(int j=1;j<NCELLJ; j++)
        {
            for(int i=1;i<NCELLI; i++)
            {
                const size_t p = ID(j,i);
                T[p] = SP[p];
            }
        }
        return;
    }
    
    APPLYIC_CG();

    APPLYBC_TEMP_CG();

    CALC_RESIDUAL_CG();

    SOLVER_CG();

    for(int j=1;j<NCELLJ; j++)
    {
        for(int i=1;i<NCELLI;i++)
        {
            const size_t p = ID(j,i);
            T[p] = TCG[p];
        }
    }
}

double UPDATE_TRANSIENT()
{
    const int ni = NI, ncelli = NCELLI, ncellj = NCELLJ;

    const double* RESTRICT t    = T.data();
    const double* RESTRICT told = T_old_time.data();
    const double invdt = 1.0/deltaT;

    double sumsq = 0.0;

    for(int j=1; j<ncellj; j++)
    {
        const size_t row = (size_t)j*ni;

        for(int i=1; i<ncelli; i++)
        {
            const size_t p = row + i;
            const double d = (t[p] - told[p])*invdt;
            sumsq = sumsq + d*d;
        }
    }

    //swap now lives at the TOP of the timestep loop in main() -- keeping it here
    //left T holding level n, so every written file lagged one step behind its label.
    return sqrt(sumsq/((double)(NI-2)*(double)(NJ-2)));
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

    //I and J count CELL CORNERS, not solution nodes: NCX+1 by NCY+1 corners enclose
    //NCX by NCY full-size cells. X and Y are written for every corner, T for every
    //cell, flagged CELLCENTERED. DATAPACKING=BLOCK is mandatory whenever VARLOCATION
    //is used; F=POINT silently produces garbage. The index in ([3]=CELLCENTERED) is
    //1-based, so 3 is T.
    const int NCX = NI-2;
    const int NCY = NJ-2;

    out << "TITLE = \"2D Transient Heat Conduction (cell-centred FVM)\"" << endl;
    out << "VARIABLES = \"X\", \"Y\", \"T\"" << endl;

    out << "ZONE T=\"t=" << simTime << "\""
        << ", I=" << NCX+1 << ", J=" << NCY+1
        << ", DATAPACKING=BLOCK, VARLOCATION=([3]=CELLCENTERED)"
        << ", STRANDID=1, SOLUTIONTIME=" << simTime << endl;

    for (int j = 0; j <= NCY; j++)
        for (int i = 0; i <= NCX; i++) out << (double)i*DELX << endl;

    for (int j = 0; j <= NCY; j++)
        for (int i = 0; i <= NCX; i++) out << (double)j*DELY << endl;

    for (int j = 1; j < NCELLJ; j++)
        for (int i = 1; i < NCELLI; i++) out << T[ID(j,i)] << endl;

    out.close();
}

void WRITE_FILE_TRANSIENT_VTK()
{
    //Numbered so ParaView picks the files up as a time series automatically.
    char fname[512];
    snprintf(fname, sizeof(fname), OUTPUT_DIR "/temperature_%05d.vtk", TIMESTEP);

    ofstream out(fname);
    if(!out.is_open())
    {
        cout << "ERROR: could not open " << fname << " for writing." << endl;
        exit(1);
    }

    out << fixed << setprecision(8);

    //CELL-CENTRED OUTPUT. The grid written here is the set of CELL CORNERS
    //(0, DELX, 2*DELX, ... LX): NCX+1 by NCY+1 points enclosing NCX by NCY full-size
    //cells, with T written as CELL_DATA. Writing the SOLUTION NODES as grid points
    //instead produces half-width slivers along every wall, because a boundary node
    //sits only DELX/2 from the first cell centre. Those slivers are a rendering
    //artefact of point data on a cell-centred mesh, not a mesh defect: every real
    //control volume is DELX by DELY.
    //
    //RECTILINEAR_GRID rather than STRUCTURED_GRID because a uniform Cartesian mesh
    //needs only three coordinate axes instead of NCX*NCY explicit point triples.
    //Switch back to STRUCTURED_GRID when the mesh becomes curvilinear.
    //
    //DIMENSIONS counts POINTS, so NCX+1 yields NCX cells. CELL_DATA must be exactly
    //NCX*NCY or ParaView rejects the file.
    const int NCX = NI-2;
    const int NCY = NJ-2;

    out << "# vtk DataFile Version 3.0" << endl;
    out << "2D Transient Heat Conduction (cell-centred FVM), t = " << simTime << endl;
    out << "ASCII" << endl;
    out << "DATASET RECTILINEAR_GRID" << endl;
    out << "DIMENSIONS " << NCX+1 << " " << NCY+1 << " " << 1 << endl;

    out << "X_COORDINATES " << NCX+1 << " double" << endl;
    for (int i = 0; i <= NCX; i++) out << (double)i*DELX << endl;

    out << "Y_COORDINATES " << NCY+1 << " double" << endl;
    for (int j = 0; j <= NCY; j++) out << (double)j*DELY << endl;

    out << "Z_COORDINATES 1 double" << endl;
    out << 0.0 << endl;

    out << "CELL_DATA " << NCX*NCY << endl;
    out << "SCALARS Temperature double 1" << endl;
    out << "LOOKUP_TABLE default" << endl;

    for (int j = 1; j < NCELLJ; j++)
    {
        for (int i = 1; i < NCELLI; i++)
        {
            out << T[ID(j,i)] << endl;
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

void CHECK_FLUX_BALANCE()
{
    const double dWb = XC[1]      - XC[0];
    const double dEb = XC[NI-1]   - XC[NI-2];
    const double dSb = YC[1]      - YC[0];
    const double dNb = YC[NJ-1]   - YC[NJ-2];

    double Qw = 0.0, Qe = 0.0, Qs = 0.0, Qn = 0.0;

    for(int j=1;j<NCELLJ;j++)
    {
        Qw += kT*AFX*(T[ID(j,0)]      - T[ID(j,1)])/dWb;
        Qe += kT*AFX*(T[ID(j,NI-1)]   - T[ID(j,NI-2)])/dEb;
    }

    for(int i=1;i<NCELLI;i++)
    {
        Qs += kT*AFY*(T[ID(0,i)]      - T[ID(1,i)])/dSb;
        Qn += kT*AFY*(T[ID(NJ-1,i)]   - T[ID(NJ-2,i)])/dNb;
    }

    double Qgen = 0.0;

    for(int j=1; j<NCELLJ; j++)
    {
        for(int i=1; i<NCELLI;i++)
        {
            const size_t p = ID(j,i);
            Qgen += (QVOL[p]+QSLOPE[p]*T[p])*VOL;
        }
    }

    const double net   = Qw + Qe + Qs + Qn + Qgen;
    const double scale = fabs(Qw)+fabs(Qe)+fabs(Qs)+fabs(Qn) + fabs(Qgen);

    cout << "BALANCE : Qw = " << scientific << setprecision(4) << Qw
         << "   Qe = " << Qe << "   Qs = " << Qs << "   Qn = " << Qn << endl;
    cout << "          net = " << net
         << "   normalised = " << (scale > 0.0 ? net/scale : 0.0)
         << "   (-> 0 at steady state)" << endl;
    cout << "          Qgen = " << Qgen << "   net = " << net
        << "   normalised = " << (scale > 0.0 ? net/scale : 0.0)
        << "   (-> 0 at steady state)" << endl;
}

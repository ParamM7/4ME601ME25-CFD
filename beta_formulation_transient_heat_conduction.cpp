#include <iostream>
#include <math.h>
#include <stdio.h>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#if defined(_WIN32)
    #include <direct.h>
    #define MAKE_DIR(p) _mkdir(p)
#else
    #include <sys/stat.h>
    #define MAKE_DIR(p) mkdir(p, 0755)
#endif

using namespace std;

#define NI 500
#define NJ 100

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

double BETA = 1.0;

#define OUTPUT_DIR "results"

#define BETA_ZERO_TOL 1.0e-14

bool VERBOSE = false;

double XCELL[2][NJ][NI];

double DELX, DELY;

double SP[NJ][NI];

int i, j, k, l, ITER;
int NCELLI;
int NCELLJ;

double LX,LY;

double T[NJ][NI];
double T_old_iter[NJ][NI];
double T_old_time[NJ][NI];

double CCSS; //convergence criteria

double AW[NJ][NI], AS[NJ][NI], AE[NJ][NI], AN[NJ][NI];
double AP[NJ][NI];

double AW1[NJ][NI], AS1[NJ][NI], AE1[NJ][NI], AN1[NJ][NI];
double AP1[NJ][NI];

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

double CALC_NORM_L2(double ARR[NJ][NI])
{
    int ii, jj;
    
    double sumsq = 0.0;

    for(jj=1;jj<NCELLJ;jj++)
    {
        for(ii=1;ii<NCELLI;ii++)
        {
            sumsq = sumsq + ARR[jj][ii]*ARR[jj][ii];
        }
    }

    return sqrt(sumsq/((double)(NI-2)*(double)(NJ-2)));
}

double CALC_NORM_L1(double ARR[NJ][NI])
{
    int ii, jj;
    
    double sumabs = 0.0;

    for(jj=1;jj<NCELLJ;jj++)
    {
        for(ii=1;ii<NCELLI;ii++)
        {
            sumabs = sumabs + fabs(ARR[jj][ii]);
        }
    }

    return sumabs/((double)(NI-2)*(double)(NJ-2));
}

//CONJUGATE GRADIENT

#define MAXITER_CG 50000

double TCG[NJ][NI]; //CG gradient solution field

double RES[NJ][NI]; //CG residual vector, r

double PDIR[NJ][NI]; //Search direction, p

double AP_CG[NJ][NI]; //A*p

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

void MAKE_OUTPUT_DIR();

int main(int argc, char* argv[]){
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
    
    MAKE_OUTPUT_DIR();

    SET_GEOMETRY();

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
        SOLVER_TRANSIENT();

        APPLYBC_TEMP_TRANSIENT();

        double dTdt = UPDATE_TRANSIENT();

        simTime = simTime + deltaT;
        TIMESTEP = TIMESTEP + 1;
        
        if(TIMESTEP % 1 == 0)
        {
            cout << "STEP = " << setw(6) << TIMESTEP
                 << "   t = " << fixed << setprecision(6) << simTime
                 << "   ||dT/dt|| = " << scientific << setprecision(4) << dTdt
                 << endl;
        }

        if(simTime + 0.5*deltaT >= nextWriteTime)
        {
            WRITE_FILE_TRANSIENT();
            WRITE_FILE_TRANSIENT_VTK();
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

    if(BETA >= BETA_ZERO_TOL)
    {
        cout << "Total CG iterations = " << TOTAL_CG_ITER
            << "   (average " << (double)TOTAL_CG_ITER/(double)(TIMESTEP>0?TIMESTEP:1)
            << " per timestep)" << endl;
    }

    
    return (0);
    
}

void SET_GEOMETRY()
{
    DELX = LX/(NI-1);
    DELY = LY/(NJ-1);

    for(j=0; j<NJ; j++){
       for(i=0;i<NI;i++){
        XCELL[0][j][i] = DELX*i;
        XCELL[1][j][i] = DELY*j;

       }
     }
}

void CALC_COEFF()
{
    for(j=1;j<NCELLJ;j++)
    {
        for(i=1;i<NCELLI;i++)
        {
            AW[j][i] = 1.0/(DELX*DELX);
            AE[j][i] = 1.0/(DELX*DELX);
            AS[j][i] = 1.0/(DELY*DELY);
            AN[j][i] = 1.0/(DELY*DELY);

            AP[j][i] = AW[j][i]+AE[j][i]+AS[j][i]+AN[j][i];
        }
    }
}


//CONJUGATE GRADIENT

void APPLYIC_CG()
{
    for(j=0;j<NJ;j++)
    {
        for(i=0;i<NI;i++)
        {
            TCG[j][i] = T_old_time[j][i];
            RES[j][i] = 0.0;
            PDIR[j][i] = 0.0;
            AP_CG[j][i] = 0.0;
        }
    }
}

void APPLYBC_TEMP_CG()
{
    //using generalized boundary condition c+a(dT/dn)=bT
    double a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;

    //west
    a1 = A_WEST;b1=B_WEST;c1=C_WEST;
    for(j=0;j<NJ;j++)
    {
       TCG[j][0] = (a1*TCG[j][1]+c1*DELX)/(a1+b1*DELX);
    }
    //south
    a2 = A_SOUTH;b2=B_SOUTH;c2=C_SOUTH;
    for(i=1;i<NCELLI;i++)
    {       
        TCG[0][i] = (a2*TCG[1][i]+c2*DELY)/(a2+b2*DELY);
    }
    //east
    a3 = A_EAST;b3=B_EAST;c3=C_EAST;
    for(j=0;j<NJ;j++)
    {        
        TCG[j][NCELLI] = (a3*TCG[j][NCELLI-1]+c3*DELX)/(a3+b3*DELX);
    }
    //north
    a4 = A_NORTH;b4=B_NORTH;c4=C_NORTH;
    for(i=1;i<NCELLI;i++)
    {        
        TCG[NCELLJ][i] = (a4*TCG[NCELLJ-1][i]+c4*DELY)/(a4+b4*DELY);
    }

}

void CALC_RESIDUAL_CG()
{
    RS_OLD = 0.0;

    for(j=1;j<NCELLJ;j++)
    {
        for(i=1;i<NCELLI;i++)
        {
            double Ax = AP[j][i]*TCG[j][i]
                        - AW[j][i]*TCG[j][i-1] - AE[j][i]*TCG[j][i+1]
                        - AS[j][i]*TCG[j-1][i] - AN[j][i]*TCG[j+1][i];

            RES[j][i] = SP[j][i] - Ax; //r = b-Ax
            PDIR[j][i] = RES[j][i]; //p0 = r0

            RS_OLD = RS_OLD + RES[j][i]*RES[j][i]; //r.r
        }
    }
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
    RRMS_CG = CALC_NORM(RES);
    if(RRMS_CG < CCSS)
    {
        ITER_CG = 0;
        return;
    }

    for(ITER_CG=1; ITER_CG<=MAXITER_CG; ITER_CG++)
    {
        // Ap = A*p
        //Matrix-free: A is never stored, only applied as a 5-point stencil. PDIR is
        //zero on the boundary (APPLYIC_CG zeroes the full field and only interior
        //nodes are ever written), so those neighbour terms drop out cleanly -- which
        //is exactly the homogeneous-Dirichlet operator CG needs.

        for(j=1;j<NCELLJ;j++)
        {
            for(i=1;i<NCELLI;i++)
            {
                AP_CG[j][i] = AP[j][i]*PDIR[j][i]
                            - AW[j][i]*PDIR[j][i-1] - AE[j][i]*PDIR[j][i+1]
                            - AS[j][i]*PDIR[j-1][i] - AN[j][i]*PDIR[j+1][i];
            }
        }

        //ALPHA_CG = (r.r)/(p*Ap)
        PAP_CG = 0.0;
        for(j=1;j<NCELLJ;j++)
        {
            for(i=1;i<NCELLI;i++)
            {
                PAP_CG = PAP_CG + PDIR[j][i]*AP_CG[j][i];
            }
        }

        //For an SPD matrix p.Ap > 0 strictly. Zero means either convergence to machine
        //precision or a broken (non-symmetric) operator -- stop either way.
        if(fabs(PAP_CG) < 1.0e-30)
        {
            break;
        }

        ALPHA_CG = RS_OLD/PAP_CG;

        //update solution, x and residual, r simultaneously
        //RS_NEW = r_new.r_new
        //The recurrence r <- r - alpha*Ap avoids recomputing b - Ax each iteration,
        //saving one full stencil sweep per iteration.
        RS_NEW = 0.0;
        for(j=1;j<NCELLJ;j++)
        {
            for(i=1;i<NCELLI;i++)
            {
                TCG[j][i] = TCG[j][i] + ALPHA_CG*PDIR[j][i];
                RES[j][i] = RES[j][i] - ALPHA_CG*AP_CG[j][i];

                RS_NEW = RS_NEW + RES[j][i]*RES[j][i];
            }
        }

        RRMS_CG = CALC_NORM(RES);

        if(RRMS_CG < CCSS)
        {
            break;
        }

        //BETA_CG to update PDIR.
        //beta is what keeps successive directions A-conjugate; drop it and this
        //degenerates into steepest descent.

        BETA_CG = RS_NEW/RS_OLD;

        for(j=1;j<NCELLJ;j++)
        {
            for(i=1;i<NCELLI;i++)
            {
                PDIR[j][i] = RES[j][i] + BETA_CG*PDIR[j][i];
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

    for (j = 0; j < NJ; j++)
    {
        for (i = 0; i < NI; i++)
        {
            out << XCELL[0][j][i] << " " << XCELL[1][j][i] << " " << TCG[j][i] << endl;
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

    for (j = 0; j < NJ; j++)
    {
        for (i = 0; i < NI; i++)
        {
            out << XCELL[0][j][i] << " " << XCELL[1][j][i] << " " << 0.0 << endl;
        }
    }

    out << "POINT_DATA " << NI*NJ << endl;
    out << "SCALARS Temperature double 1" << endl;
    out << "LOOKUP_TABLE default" << endl;

    for (j = 0; j < NJ; j++)
    {
        for (i = 0; i < NI; i++)
        {
            out << TCG[j][i] << endl;
        }
    }

    out.close();

    cout << "Field written to temperature_field_CG.vtk (ParaView Legacy VTK)" << endl;
}

//define the transient heat conduction problem with explicit time marching scheme

void APPLYIC_TRANSIENT()
{
    for(j=0;j<NJ;j++)
    {
        for(i=0;i<NI;i++)
        {
            T[j][i] = T_INIT;
            T_old_iter[j][i] = T_INIT;
            T_old_time[j][i] = T_INIT;
        }
    }
}

void APPLYBC_TEMP_TRANSIENT()
{
    double a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;

    //west
    a1 = A_WEST;b1=B_WEST;c1=C_WEST;
    for(j=0;j<NJ;j++)
    {
        T[j][0] = (a1*T[j][1]+c1*DELX)/(a1+b1*DELX);
        T_old_time[j][0] = (a1*T_old_time[j][1]+c1*DELX)/(a1+b1*DELX);
    }
    //south
    a2 = A_SOUTH;b2=B_SOUTH;c2=C_SOUTH;
    for(i=1;i<NCELLI;i++)
    {
        T[0][i] = (a2*T[1][i]+c2*DELY)/(a2+b2*DELY);
        T_old_time[0][i] = (a2*T_old_time[1][i]+c2*DELY)/(a2+b2*DELY);
    }
    //east
    a3 = A_EAST;b3=B_EAST;c3=C_EAST;
    for(j=0;j<NJ;j++)
    {
        T[j][NCELLI] = (a3*T[j][NCELLI-1]+c3*DELX)/(a3+b3*DELX);
        T_old_time[j][NCELLI] = (a3*T_old_time[j][NCELLI-1]+c3*DELX)/(a3+b3*DELX);
    }
    //north
    a4 = A_NORTH;b4=B_NORTH;c4=C_NORTH;
    for(i=1;i<NCELLI;i++)
    {
        T[NCELLJ][i] = (a4*T[NCELLJ-1][i]+c4*DELY)/(a4+b4*DELY);
        T_old_time[NCELLJ][i] = (a4*T_old_time[NCELLJ-1][i]+c4*DELY)/(a4+b4*DELY);
    }
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
    for(j=1;j<NCELLJ;j++)
    {
        for(i=1;i<NCELLI;i++)
        {
            AW[j][i] = BETA*Fo_x;
            AE[j][i] = BETA*Fo_x;
            AN[j][i] = BETA*Fo_y;
            AS[j][i] = BETA*Fo_y;

            AP[j][i] = 1.0 + 2.0*BETA*(Fo_x + Fo_y);

            AW1[j][i] = (1.0-BETA)*Fo_x;
            AE1[j][i] = (1.0-BETA)*Fo_x;
            AN1[j][i] = (1.0-BETA)*Fo_y;
            AS1[j][i] = (1.0-BETA)*Fo_y;

            AP1[j][i] = 1.0 - 2.0*(1.0-BETA)*(Fo_x + Fo_y);
        }
    }

    if(AP1[1][1] < 0.0)
    {
        cout << "WARNING : AP1 = " << fixed << setprecision(4) << AP1[1][1]
             << " < 0 violates Patankar's positive-coefficient rule. The scheme is"
             << " stable but the solution may oscillate; non-oscillatory behaviour"
             << " needs Fo_x + Fo_y <= 1/(2*(1-BETA))." << endl;
    }
}

void BUILD_RHS_TRANSIENT()
{
    for(j=1;j<NCELLJ;j++)
    {
        for(i=1;i<NCELLI;i++)
        {
            SP[j][i] = AP1[j][i]*T_old_time[j][i]
                     + AW1[j][i]*T_old_time[j][i-1] + AE1[j][i]*T_old_time[j][i+1]
                     + AS1[j][i]*T_old_time[j-1][i] + AN1[j][i]*T_old_time[j+1][i];
        }
    }
}

void SOLVER_TRANSIENT()
{
    BUILD_RHS_TRANSIENT();

    if(BETA < BETA_ZERO_TOL)
    {
        for(j=1;j<NCELLJ; j++)
        {
            for(i=1;i<NCELLI; i++)
            {
                T[j][i] = SP[j][i];
            }
        }
        return;
    }
    
    APPLYIC_CG();

    APPLYBC_TEMP_CG();

    CALC_RESIDUAL_CG();

    SOLVER_CG();

    for(j=1;j<NCELLJ; j++)
    {
        for(i=1;i<NCELLI;i++)
        {
            T[j][i] = TCG[j][i];
        }
    }
}

double UPDATE_TRANSIENT()
{
    double sumsq = 0.0;

    for(j=1;j<NCELLJ;j++)
    {
        for(i=1;i<NCELLI;i++)
        {
            double d = (T[j][i] - T_old_time[j][i])/deltaT;
            sumsq = sumsq + d*d;
        }
    }

    for(j=0;j<NJ;j++)
    {
        for(i=0;i<NI;i++)
        {
            T_old_time[j][i] = T[j][i];
        }
    }
    
    return sqrt(sumsq/((NI-2)*(NJ-2)));
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

    for (j = 0; j < NJ; j++)
    {
        for (i = 0; i < NI; i++)
        {
            out << XCELL[0][j][i] << " " << XCELL[1][j][i] << " " << T[j][i] << endl;
        }
    }

    out.close();
}

void WRITE_FILE_TRANSIENT_VTK()
{
    //Same STRUCTURED_GRID reasoning as WRITE_FILE_CG_VTK() above; numbered so ParaView
    //picks the files up as a time series automatically.
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

    for (j = 0; j < NJ; j++)
    {
        for (i = 0; i < NI; i++)
        {
            out << XCELL[0][j][i] << " " << XCELL[1][j][i] << " " << 0.0 << endl;
        }
    }

    out << "POINT_DATA " << NI*NJ << endl;
    out << "SCALARS Temperature double 1" << endl;
    out << "LOOKUP_TABLE default" << endl;

    for (j = 0; j < NJ; j++)
    {
        for (i = 0; i < NI; i++)
        {
            out << T[j][i] << endl;
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
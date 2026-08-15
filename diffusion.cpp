#include <iostream>
#include <math.h>
#include <stdio.h>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define NI 1000
#define NJ 1000

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

double XCELL[2][NJ][NI];

double DELX, DELY;

double SP[NJ][NI];

int i, j, k, l, ITER;
int NCELLI;
int NCELLJ;

double LX,LY;

double T[NJ][NI];
double T_old[NJ][NI];

double CCSS; //convergence criteria

double AW[NJ][NI], AS[NJ][NI], AE[NJ][NI], AN[NJ][NI];
double AP[NJ][NI];

double RRMS, RMSRESIDUE;

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
    double sumsq = 0.0;

    for(j=1;j<NCELLJ;j++)
    {
        for(i=1;i<NCELLI;i++)
        {
            sumsq = sumsq + ARR[j][i]*ARR[j][i];
        }
    }

    return sqrt(sumsq/((NI-2)*(NJ-2)));
}

double CALC_NORM_L1(double ARR[NJ][NI])
{
    double sumabs = 0.0;

    for(j=1;j<NCELLJ;j++)
    {
        for(i=1;i<NCELLI;i++)
        {
            sumabs = sumabs + fabs(ARR[j][i]);
        }
    }

    return sumabs/((NI-2)*(NJ-2));
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

#define kT 1.0 //thermal conductivity
#define rho 1.0 //density
#define Cp 1.0 //specific heat

double deltaT = 0.0001;

double alpha = kT/(rho*Cp);

double Fo;

int main(){
    CCSS = 1e-6;
    LX = 1.0;
    LY = 1.0;

    NCELLI = NI-1;
    NCELLJ = NJ-1;

    SET_GEOMETRY();

    //APPLYIC();

    CALC_COEFF();
    ITER = 0;

    //ITERATION:

    //APPLYBC_TEMP();
    //RRMS=0;
    //SOLVER_GS();

    /*
    if(RMSRESIDUE>CCSS){
        UPDATE();
        ITER = ITER+1;
        goto ITERATION;
    }
    */
    //cout<<"No of iterations taken: = "<<ITER<<endl;

    //WRITE_FILE();

    //CONJUGATE GRADIENT

    APPLYIC_CG();
    APPLYBC_TEMP_CG();
    CALC_RESIDUAL_CG();
    SOLVER_CG();

    cout<<"Conjugate Gradient: no of iterations taken: = "<<ITER_CG<<endl;

    WRITE_FILE_CG();
    WRITE_FILE_CG_VTK();
    return (0);
    
}

void SET_GEOMETRY()
{
    DELX = LX/(NI-1);
    DELY = LY/(NJ-1);

    for(j=0; j<NJ; j++){
       for(i=0;i<NI;i++){
        XCELL[0][j][i] = DELX*i;
        XCELL[1][j][i] = DELX*j;

       }
     }
}

void APPLYIC()
{
    for(j=0;j<NJ;j++)
    {
        for(i=0;i<NI;i++)
        {
            T[j][i] = 10.0;
            T_old[j][i] = 10.0;
        }
    }
}

void APPLYBC_TEMP()
{
    //using generalized boundary condition c+a(dT/dn)=bT
    double a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;

    //west
    a1 = 0;b1=1;c1=10;
    for(j=0;j<NJ;j++)
    {
        T_old[j][0] = (a1*T_old[j][1]+c1*DELX)/(a1+b1*DELX);
        T[j][0] = (a1*T[j][1]+c1*DELX)/(a1+b1*DELX);
    }
    //south
    a2 = 0;b2=1;c2=20;
    for(i=1;i<NCELLI;i++)
    {
        T_old[0][i] = (a2*T_old[1][i]+c2*DELY)/(a2+b2*DELY);
        T[0][i] = (a2*T[1][i]+c2*DELY)/(a2+b2*DELY);
    }
    //east
    a3 = 0;b3=1;c3=30;
    for(j=0;j<NJ;j++)
    {
        T_old[j][NCELLI] = (a3*T_old[j][NCELLI-1]+c3*DELX)/(a3+b3*DELX);
        T[j][NCELLI] = (a3*T[j][NCELLI-1]+c3*DELX)/(a3+b3*DELX);
    }
    //north
    a4 = 0;b4=1;c4=40;
    for(i=1;i<NCELLI;i++)
    {
        T_old[NCELLJ][i] = (a4*T_old[NCELLJ-1][i]+c4*DELY)/(a4+b4*DELY);
        T[NCELLJ][i] = (a4*T[NCELLJ-1][i]+c4*DELY)/(a4+b4*DELY);
    }

}

void UPDATE()
{
    for(j=0;j<NJ;j++)
    {
        for(i=0;i<NI;i++)
        {
            T_old[j][i] = T[j][i];

            SP[j][i] = 0.0; //no volumetric source -> pure Laplace
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

void SOLVER_GS()
{

    for(j=1;j<NCELLJ;j++){

        for(i=1; i< NCELLI; i++){
            
            T[j][i] = (AW[j][i]*T[j][i-1]+AE[j][i]*T_old[j][i+1]+AS[j][i]*T[j-1][i]+AN[j][i]*T_old[j+1][i]+SP[j][i])/(AP[j][i]);

            RRMS = RRMS + (T[j][i]-T_old[j][i])*(T[j][i]-T_old[j][i]);
        }
    }

    RMSRESIDUE = sqrt(RRMS/((NI-2)*(NJ-2)));
    if(ITER % 50 == 0)
    {
        cout << "ITER = " << ITER << "   RMSRESIDUE = " << RMSRESIDUE << endl;
    }
}

void WRITE_FILE()
{
    ofstream out("temperature_field.dat");
    out << fixed << setprecision(8);

    out << "TITLE = \"2D Steady-State Heat Diffusion (Gauss-Seidel FDM)\"" << endl;
    out << "VARIABLES = \"X\", \"Y\", \"T\"" << endl;
    out << "ZONE T=\"Steady Heat Diffusion\", I=" << NI << ", J=" << NJ
        << ", F=POINT" << endl;

    for (j = 0; j < NJ; j++)
    {
        for (i = 0; i < NI; i++)
        {
            out << XCELL[0][j][i] << " " << XCELL[1][j][i] << " " << T[j][i] << endl;
        }
    }

    out.close();

    cout << "Field written to temperature_field.dat (Tecplot ASCII)" << endl;
}

//CONJUGATE GRADIENT

void APPLYIC_CG()
{
    for(j=0;j<NJ;j++)
    {
        for(i=0;i<NI;i++)
        {
            TCG[j][i] = 10.0;
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
    a1 = 10;b1=5;c1=0;
    for(j=0;j<NJ;j++)
    {
       TCG[j][0] = (a1*TCG[j][1]+c1*DELX)/(a1+b1*DELX);
    }
    //south
    a2 = 0;b2=1;c2=0;
    for(i=1;i<NCELLI;i++)
    {       
        TCG[0][i] = (a2*TCG[1][i]+c2*DELY)/(a2+b2*DELY);
    }
    //east
    a3 = 0;b3=1;c3=0;
    for(j=0;j<NJ;j++)
    {        
        TCG[j][NCELLI] = (a3*TCG[j][NCELLI-1]+c3*DELX)/(a3+b3*DELX);
    }
    //north
    a4 = 0;b4=1;c4=0;
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
    for(ITER_CG=1; ITER_CG<=MAXITER_CG; ITER_CG++)
    {
        // Ap = A*p

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
        ALPHA_CG = RS_OLD/PAP_CG;

        //update solution, x and residual,r simulatneously
        //RS_NEW = r_new.r_new
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

        if(ITER_CG % 50 == 0)
        {
            cout << "ITER_CG = " << ITER_CG << "   RRMS_CG = " << RRMS_CG << endl;
        }

        if(RRMS_CG < CCSS)
        {
            break;
        }

        //BETA_CG to update PDIR

        BETA_CG = RS_NEW/RS_OLD;

        for(j=1;j<NCELLJ;j++)
        {
            for(i=1;i<NCELLI;i++)
            {
                PDIR[j][i] = RES[j][i] + BETA_CG*PDIR[j][i];
            }
        }

        RS_OLD = RS_NEW;
    }
}

void WRITE_FILE_CG()
{
    ofstream out("temperature_field_CG.dat");
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
// Writes the CG solution as a Legacy VTK ASCII file, which ParaView opens
// directly (File > Open, pick the .vtk, click Apply) -- no import
// settings or format guessing needed, unlike the Tecplot .dat file.
//
// DATASET STRUCTURED_GRID is used rather than the simpler
// STRUCTURED_POINTS/ImageData type: STRUCTURED_POINTS assumes a strictly
// uniform, axis-aligned spacing (ORIGIN + fixed SPACING only), whereas
// STRUCTURED_GRID takes an explicit POINTS list -- so this keeps working
// unchanged even if XCELL is ever replaced with a stretched or
// non-uniform grid later.
//
// VTK legacy format requirements this follows:
//   - DIMENSIONS nx ny nz -- VTK always thinks in 3D, so a 2D field is
//     just "one cell thick" in z (nz = 1).
//   - POINTS: one x,y,z triple per grid point, with the X-index varying
//     FASTEST, then Y, then Z. That's exactly the i-inner/j-outer loop
//     order already used by every other WRITE_* function in this file --
//     no reordering needed, z is simply written as 0.0 throughout.
//   - POINT_DATA: one scalar per point, listed in the SAME order as
//     POINTS above -- here the temperature field TCG.
{
    ofstream out("temperature_field_CG.vtk");
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

#include "Eigen/Eigen" //internal library
#include <iostream>
#include <limits>
#include <iomanip>

using namespace Eigen;
using namespace std;

/* Legend of the script*/
/* 1)Function that checks if the Matrix (A) of the linear system is singular
 * 2-3)Functions that calculates PALU and QR factorizations of A
 * 4)Function that evaluates the relative error of the solution with what obtained in 2) and 3)
 * 5)Main, where input are given and some ifs prevent singularity issues before recalling 2) and 3)
 * */
bool singularitycheck(const Matrix2d& A)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d svA = svd.singularValues();
    if(svA.minCoeff() < numeric_limits<double>::epsilon()) //eigenvalue = 0 ==> A is singular ==> no factorizations are allowed
    {
        return false;
    }
    return true;
}



Vector2d solvePALU(const Matrix2d& A,const Vector2d&  b)
{
    Vector2d xPALU = A.fullPivLu().solve(b); //solve with PALU factorization with full pivoting
    return xPALU;
}

Vector2d solveQR(const Matrix2d& A, const Vector2d& b)
{
    Vector2d xQR = A.colPivHouseholderQr().solve(b); //solve with QR factorization with column pivoting maximizing stability
    return xQR;
}


double err(const Vector2d& x_exact, const Vector2d& x_calculated) //relative error's function
{
    double errRel = (x_exact - x_calculated).norm() / x_exact.norm();
    return errRel;
}

int main()
{
    Vector2d x_exact;
    x_exact << -1.0e+0,-1.0e+00;

    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01,-9.992887623566787e-01;
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    Matrix2d A2;
    A2 << 5.547001962252291e-01,-5.540607316466765e-01, 8.320502943378437e-01,-8.324762492991313e-01;
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    Matrix2d A3;
    A3 << 5.547001962252291e-01,-5.547001955851905e-01, 8.320502943378437e-01,-8.320502947645361e-01;
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    //on A1,b1
    if (!singularitycheck(A1))
    {
        cout << "la matrice A1 e' singolare" << endl; //if singular no factorization are calculated
    }
    else
    {
        Vector2d xPALU1 = solvePALU(A1,b1);
        Vector2d xQR1 = solveQR(A1,b1);
        cout << "L'errore relativo per il sistema A1=xb1 con la fattorizzazione PALU e': " << scientific << setprecision(16) << err(x_exact,xPALU1) << endl
             <<"L'errore relativo per il sistema A1=xb1 con la fattorizzazione Qr e': " << scientific << setprecision(16) << err(x_exact,xQR1) << endl;
    }

    //on A2,b2
    if (!singularitycheck(A2))
    {
        cout << "la matrice A2 e' singolare" << endl; //if singular no factorization are calculated
    }
    else
    {
        Vector2d xPALU2 = solvePALU(A2,b2);
        Vector2d xQR2 = solveQR(A2,b2);
        cout << "L'errore relativo per il sistema A2=xb2 con la fattorizzazione PALU e': " << scientific << setprecision(16)<< err(x_exact,xPALU2) << endl
             <<"L'errore relativo per il sistema A2=xb2 con la fattorizzazione Qr e': " << scientific << setprecision(16) << err(x_exact,xQR2) << endl;
    }

    //on A3,b3
    if (!singularitycheck(A3))
    {
        cout << "la matrice A3 e' singolare" << endl; //if singular no factorization are calculated
    }
    else
    {
        Vector2d xPALU3 = solvePALU(A3,b3);
        Vector2d xQR3 = solveQR(A3,b3);
        cout << "L'errore relativo per il sistema A3=xb3 con la fattorizzazione PALU e': " << scientific << setprecision(16) << err(x_exact,xPALU3) << endl
             <<"L'errore relativo per il sistema A3=xb3 con la fattorizzazione Qr e': " << scientific << setprecision(16) << err(x_exact,xQR3) << endl;
    }


    return 0;
}

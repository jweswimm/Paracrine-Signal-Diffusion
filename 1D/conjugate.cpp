#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

const double NEARZERO = 1.0e-10;       // interpretation of "zero"


//Initialize Functions for Conjugate Gradient Descent
std::vector<double> matrixTimesVector( const std::vector<std::vector<double>> &A, const std::vector<double> &V );
std::vector<double> vectorCombination( double a, const std::vector<double> &U, double b, const std::vector<double> &V );
double innerProduct( const std::vector<double> &U, const std::vector<double> &V );
double vectorNorm( const std::vector<double> &V );
std::vector<double> conjugateGradientSolver( const std::vector<std::vector<double>> &A, const std::vector<double> &B );
void printVector(std::vector <double> const &a);


//======================================================================


int main()
{
   std::vector<std::vector<double>> A = { { 1, 0.5, 0 }, { 0.5, 1, 0 }, {0,0,1} };
   std::vector<double> B = { 1, 2 ,3};

   std::vector<double> X = conjugateGradientSolver( A, B );
   printVector(X);
}


//======================================================================


std::vector<double> matrixTimesVector( const std::vector<std::vector<double>> &A, const std::vector<double> &V )     // Matrix times vector
{
    //solves A*V=C (return C)
   int n = A.size(); 
   std::vector<double> C( n );
   for ( int i = 0; i < n; i++ ) {
   C[i] = innerProduct( A[i], V );
   }
   return C;
}


//======================================================================


std::vector<double> vectorCombination( double a, const std::vector<double> &U, double b, const std::vector<double> &V )        // Linear combination of vectors
{
    //aU+bV=W (return W)
   int n = U.size();
   std::vector<double> W( n );
   for ( int j = 0; j < n; j++ ){ 
    W[j] = a * U[j] + b * V[j];
}
   return W;
}


//======================================================================


double innerProduct( const std::vector<double> &U, const std::vector<double> &V )          // Inner product of U and V
{
    //stl has inner_product prebuilt function, we can just use this
   return inner_product( U.begin(), U.end(), V.begin(), 0.0 );
}


//======================================================================


double vectorNorm( const std::vector<double> &V )   // Just calculate the 2 norm (see if we use different metric)
{
   return sqrt( innerProduct( V, V ) );
}


//======================================================================


std::vector<double> conjugateGradientSolver( const std::vector<std::vector<double>> &A, const std::vector<double> &B )
{
   double TOLERANCE = 1.0e-10;

   int n = A.size();
   std::vector<double> X( n, 0.0 );

   std::vector<double> R = B;
   std::vector<double> P = R;
   int k = 0;

   while ( k < n )
   {
      std::vector<double> Rold = R;  //we will be using new R, but keep old
      std::vector<double> AP = matrixTimesVector( A, P );

      double alpha = innerProduct( R, R ) / std::max( innerProduct( P, AP ), NEARZERO ); //using max gets away from any division by 0 or negative from underfloat or computation complexities
      X = vectorCombination( 1.0, X, alpha, P );            // Find new x
      R = vectorCombination( 1.0, R, -alpha, AP );          // Calculate new residual

      if ( vectorNorm( R ) < TOLERANCE ) break;             //Did we find x?

      double beta = innerProduct( R, R ) / std::max( innerProduct( Rold, Rold ), NEARZERO );
      P = vectorCombination( 1.0, R, beta, P );             // Next gradient
      k++;
   }

   return X;
}


void printVector(std::vector <double> const &a) {
   std::cout << "The vector elements are : ";
   for(int i=0; i < a.size(); i++)
   if (a.at(i)< NEARZERO){
       std::cout<<0<< ' ';
   }
   else
   std::cout << a.at(i) << ' ';
}

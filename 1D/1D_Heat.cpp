#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

const double NEARZERO = 1.0e-10;  //to stop underflow or dividing by 0



//Initialize Functions here
std::vector<double> matrixTimesVector( const std::vector<std::vector<double>> &A, const std::vector<double> &V );
std::vector<double> vectorCombination( double a, const std::vector<double> &U, double b, const std::vector<double> &V );
double innerProduct( const std::vector<double> &U, const std::vector<double> &V );
double vectorNorm( const std::vector<double> &V );
std::vector<double> conjugateGradientSolver( const std::vector<std::vector<double>> &A, const std::vector<double> &B );
void printVector( std::string title, const std::vector<double> &V );
void printMatrix( std::string title, const std::vector<std::vector<double>> &A );

std::vector<std::vector<double>> heat_solver(const double a, const double neumann, const double IC, const double time, const double length, const double dt, const double dx);

//-------------------------------------------------------------------------



int main(){

heat_solver(5,0,4,5,5,0.01,0.01);

return 0;
}








//-----------------------------------------------------------------------

std::vector<std::vector<double>> heat_solver(const double a, const double neumann, const double IC, const double time, const double length, const double dt, const double dx){
    std::cout<<"running heat solver function" <<std::endl;
//this solves the neumann type heat equation
//u_t=a*u_xx
//du/dx(0)=du/dx(L)=neumann
//and writes the resulting solution to a text file heat_data.txt
//for now just focus on neumann=0

    std::ofstream myfile;
    myfile.open ("heat_data.txt");

    int nt=time/dt;
    int nx=length/dx; 




//initialize U and X(output) as a matrix of 0's
std::vector<std::vector<double>> U(nx,std::vector<double>(nt,0)),X(nx,std::vector<double>(nt,0));
//Start algorithm, i is x iterator, n is time iterator
//Crank-Nicolson method gives us
//-r u(i+1,n+1) + (1+2r) u(i,n+1) - r u(i-1,n+1)=r u(i+1,n) + (1 - 2r) u(i,n) + r u(i-1,n)
//where r = dt/(2 dx^2).
//The right hand side of the equation is known, so
//we note that this scheme gives a tridiagonal system, i.e. nonzero elements on diagonal and 
//directly below and above the diagonal.
//We must solve this tridiagonal system at EVERY time step.

float r = (a*dt) / (2.000 * pow(dx,2.0));
//float r = 10;
//The CN method gives us AU^(n+1)=BU^n, where A and B are matrices given in terms of r. 
//We know U^n (the array of u(i,n) values) and B (the matrix in terms of r), so we must determine
//what BU^n is through vector-matrix multiplication.

//Set up A and B matrix of 0's size nx*nt
std::vector<std::vector<double>> A(nx,std::vector<double>(nt,0)),B(nx,std::vector<double>(nt,0));

for (int i=0; i<nx; i++){
    for (int j=0; j<nx; j++){
        if (i==j && i==1){
            B[i][j]=1-r; //set B[1][1]=1-r
            B[i][j+1]=r; //set B[1][1+1]=r
            A[i][j]=1+r; //set B[1][1]=1-r
            A[i][j+1]=-r;
        }
        else if (i==j && i==nx){
            B[i][j]=1-r; //set B[nx][nx]=1-r
            B[i][j-1]=r; //set B[nx][nx-1]=r
            A[i][j]=1+r; //set B[nx][nx]=1-r
            A[i][j-1]=-r; //set B[nx][nx-1]=r
        }
        else if (i==j && i!=0){ //set diagonal = 1-2r
            B[i][j]=1-2*r;
            B[i][j-1]=r; //set off diagonal in + direction to be r
            B[i][j+1]=r; //set off diagonal in - direction to be r
            A[i][j]=1+2*r;
            A[i][j-1]=-r; //set off diagonal in + direction to be r
            A[i][j+1]=-r; //set off diagonal in - direction to be r
        }
    }
}
//printMatrix("Amatrix",A);
//printMatrix("Bmatrix",B);


//Now our tridiagonal B matrix is set up.
//Remember we are looking to set up AU^(n+2)=b, and right now we have AU^(n+1)=BU^(n). 
//Now we start our time loop, as we will be multiplying B with the known vector U^(n) at each step


//Initialize vector b
std::vector<double> b (nt);


//set U(x,t)=U(i,0)=IC
for (int j=0; j<nx; j++){
U[j][0]=IC;
myfile << U[j][0] << " ";
}
myfile << "\n";

//Initialize dummy vector dum
std::vector <double> dum(nt);
std::vector <double> dum2(nt);
//Now start are large timestep loop
std::cout<<nt<<std::endl;
std::cout<<nx<<std::endl;


for (int n=0; n<nt; n++){ //was nt
    //We are looking to find U^(n+1)
    //First determine b matrix by B * U^(n), where U^(n) is the KNOWN vector with the u(i,n) values
    for (int i=0; i<nx; i++){ 
     dum[i]=U[i][n]; //loop through space
     //std::cout<<dum[i]<<std::endl;
    }

    b=matrixTimesVector(B,dum);
    //Now we have AU=b
    //To solve for U, we enlist conjugate gradient descent
    //We can rewrite over dummy vector to save cgd output (our U^(n+1) value)
    dum2=conjugateGradientSolver(A,b);
    for (int i=0; i<nx; i++){
        U[i][n+1]=dum2[i]; //set the new U column (timestep) to the output of the cgd
        myfile << U[i][n+1] << " ";
    }
    myfile << "\n";
}
myfile.close();
printMatrix("U matrix", U);
return U;
}



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


void printVector( std::string title, const std::vector<double> &V )
{
   std::cout << title << '\n';

   int n = V.size();           
   for ( int i = 0; i < n; i++ )
   {
      double x = V[i];   if ( abs( x ) < NEARZERO ) x = 0.0;
      std::cout << x << '\t';
   }
   std::cout << '\n';
}

void printMatrix( std::string title, const std::vector<std::vector<double>> &A )
{
   std::cout << title << '\n';

   int m = A.size(), n = A[0].size();                      // A is an m x n matrix
   for ( int i = 0; i < m; i++ )
   {
      for ( int j = 0; j < n; j++ )
      {
         double x = A[i][j];   if ( abs( x ) < NEARZERO ) x = 0.0;
         std::cout << x << '\t';
      }
      std::cout << '\n';
   }
}

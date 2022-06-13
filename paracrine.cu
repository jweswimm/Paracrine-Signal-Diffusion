#include "paracrine.cuh"
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

//Paracrine Initialization Function
void paracrine::initialize(){
	std::cout << "Beginning Paracrine Initialization" << std::endl;
	//initialize x1, x1, etc. and ones
	for (int i = 0; i < nnz; i++) {
		x0.push_back(0.0);
		x1.push_back(0.0);
		y0.push_back(0.0);
		y1.push_back(0.0);
		z0.push_back(0.0);
		z1.push_back(0.0);
		ones.push_back(1.0);

	}

	//determine closest gridpoints to neurons
	//neuron_x/y/z are thrust vectors with neuron locations
	//x0,x1,etc. are vectors with the closest gridpoints (each with size nnz)
	for (int i = 0; i < nnz; i++) {
		x0[i] = floor(neuron_x[i]);
		x1[i] = ceil(neuron_x[i]);
		y0[i] = floor(neuron_y[i]);
		y1[i] = ceil(neuron_y[i]);
		z0[i] = floor(neuron_z[i]);
		z1[i] = ceil(neuron_z[i]);

	}

	//initialize Q to zeros so that we can use unique indices later
	for (int i = 0; i < nnz * 8; i++)
		Q.push_back(0.0);

//	std::cout << "the 5th neuron has position=" << neuron_y[4] << " with closest gridpoints at x0=" << y0[4] << " and x1=" << y1[4] << std::endl;


	//thrust::fill(Q.begin(), Q.end(), 100);
//	thrust::copy(x0.begin(), x0.end(), Q.begin());


	//Now calculate distance vectors
	thrust::device_vector<float> del_x(nnz);
	thrust::device_vector<float> del_y(nnz);
	thrust::device_vector<float> del_z(nnz);


	//we can create functor to apply the same operation across all of the elements of the thrust vector, but this is hard and I don't really understand it
	//see struct del_operator in paracrine.cuh
	//https://www.bu.edu/pasi/files/2011/07/Lecture6.pdf page 17 for an example of building custom thrust operations
	//for now, implement naiive way with for loop

	for (int i = 0; i < nnz; i++) {
		del_x[i] = ((neuron_x[i] - x0[i]) / (x1[i] - x0[i])); 
		del_y[i] = ((neuron_y[i] - y0[i]) / (y1[i] - y0[i]));
		del_z[i] = ((neuron_z[i] - z0[i]) / (z1[i] - z0[i]));
	}
	std::cout << "del_x:" << del_x[0] << " del_y:" << del_y[0] << " del_z:" << del_z[0] << std::endl;


	//flatten 2d thrust vectors by using https://stackoverflow.com/questions/16599501/cudamemcpy-function-usage/16616738#16616738
	//Given a matrix A[x,y], we can convert to a long vector by 
	//A[x,y]=B[rowsize*x+y]
	//Then B is a long vector that has the information of matrix A
	int rowsize = nnz;

	//build distance matrix Q
	//Q looks like [one : del_x : del_y : del_z : del_x*del_y : del_y*del_z : del_z*del_x : del_x*del_y*del_z]^T (8 x nnz)
	//but we have to flatten it, so size of Q is actually nnz*8
		for (int column = 0; column < nnz; column++) { 
			//fill out Q matrix by column so we can loop through all neurons
			//there are nnz many columns and 8 rows
			//so Q[row, column]=Q[rowsize * row + column]
			//here, rowsize=nnz,
				Q[rowsize * 0 + column] = ones[column];
				Q[rowsize * 1 + column] = del_x[column];
				Q[rowsize * 2 + column] = del_y[column];
				Q[rowsize * 3 + column] = del_z[column];
				Q[rowsize * 4 + column] = del_x[column] * del_y[column];
				Q[rowsize * 5 + column] = del_y[column] * del_z[column];
				Q[rowsize * 6 + column] = del_z[column] * del_x[column];
				Q[rowsize * 7 + column] = del_x[column] * del_y[column] * del_z[column];
		}

	//Now test to make sure Q is saved correctly
//	int column = 4;
//	for (int row = 0; row < nnz; row++)
//		std::cout << "Row: " << row << "      Q=" << Q[rowsize * row + column] << std::endl;


	//Initialize weighted spread to 0.0 so we can use unique indices
	for (int i = 0; i < nnz * 8; i++)
		weighted_spread.push_back(0.0);

	//Now create weighted_spread (used in spreading)
	//note that this is the same size as Q (8 x nnz)
		for (int column = 0; column < nnz; column++) {//[columnsize * column + row]
				weighted_spread[rowsize * 0 + column] = (ones[column]-del_x[column])*(ones[column]-del_y[column])*(ones[column]-del_z[column]);
				weighted_spread[rowsize * 1 + column] = (ones[column]-del_x[column])*(ones[column]-del_y[column])*(del_z[column]);
				weighted_spread[rowsize * 2 + column] = (ones[column]-del_x[column])*(del_y[column])*(ones[column]-del_z[column]);
				weighted_spread[rowsize * 3 + column] = (ones[column]-del_x[column])*(del_y[column])*(del_z[column]);
				weighted_spread[rowsize * 4 + column] = (del_x[column])*(ones[column]-del_y[column])*(ones[column]-del_z[column]);
				weighted_spread[rowsize * 5 + column] = (del_x[column])*(ones[column]-del_y[column])*(del_z[column]);
				weighted_spread[rowsize * 6 + column] = (del_x[column])*(del_y[column])*(ones[column]-del_z[column]);
				weighted_spread[rowsize * 7 + column] = (del_x[column])*(del_y[column])*(del_z[column]);
		}


//	std::cout << "del_x:" << del_x[0] << " del_y:" << del_y[0] << " del_z:" << del_z[0] << std::endl;
	//test weighted_spread
//	int column = 0;
//	for (int row = 0; row < nnz; row++)
//		std::cout << weighted_spread[rowsize * row + column] << std::endl; 




	//Remember that Q and weighted_spread are class members and can be used outside of this initialization function
	//they will be used in the interpolation and spreading functions respectively
}

//void paracrine::test() {
//	std::cout << "testing" << std::endl;
//}

//interpolate function
//see https://spie.org/samples/PM159.pdf for the trilinear interpolation scheme used here
//inputs: number of neurons nnz, grid_size (in 1D), concentration on grid
//returns neurotransmitter concentration at neuron locations
thrust::device_vector<float> paracrine::interpolate(int nnz, int grid_size, thrust::device_vector<float> grid) {
	thrust::device_vector<float> neuron_concentrations(nnz);

	int grid_width = grid_size; 
	int grid_height = grid_size; 
	int grid_depth = grid_size; 


	int rowsize = nnz;


	//assuming grid is already flattened (was grid_size x grid_size x grid_size, now is grid_size^3 x 1)
	//So grid[x0, y0, z0]=grid[grid_height*grid_depth*x0 + grid_depth*y0 + z0]

	//Build coefficient matrix C
	// and in the process, transpose it
	//initialize to 0
	thrust::device_vector<float> CT(nnz*8,0.0);
//	for (int i = 0; i < nnz * 8; i++)
//		C.push_back(0.0);

	//There will be c values for each element, e.g nnz many c0 values
	//Note: this must be computed every time the interpolation takes place, as the concentration
	//at the grid points will be changing(i.e. grid itself will be changing)
		for (int row = 0; row < nnz; row++) {
//			std::cout <<"testvalue: "<< rowsize * row + column << std::endl;
				CT[rowsize * row + 0] = grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

				CT[rowsize * row + 1] = grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]]  //grid[x1, y0, z0]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

				CT[rowsize * row + 2] = grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]]  //grid[x0, y1, z0]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

				CT[rowsize * row + 3] = grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]]  //grid[x0, y0, z1]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

				CT[rowsize * row + 4] = grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z0[row]] //grid[x1,y1,z0]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]]  //grid[x0,y1,z0]
					- grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]]  //grid[x1,y0,z0]
					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

				CT[rowsize * row + 5] = grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z1[row]] //grid[x0,y1,z1]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]]  //grid[x0,y0,z1]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]]  //grid[x0,y1,z0]
					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

				CT[rowsize * row + 6] = grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z1[row]] //grid[x1,y0,z1]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]]  //grid[x0,y0,z1]
					- grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]]  //grid[x1,y0,z0]
					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

				CT[rowsize * row + 7] = grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z1[row]] //grid[x1,y1,z1]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z1[row]]  //grid[x0, y1, z1]
					- grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z1[row]]  //grid[x1, y0, z1]
					- grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z0[row]]  //grid[x1, y1, z0]
					+ grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]]  //grid[x1, y0, z0]
					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]]  //grid[x0, y0, z1]
					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]]  //grid[x0, y1, z0]
					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]];  //grid[x0, y0, z0]
			


		}


	//Now that we have CT created, we have to
	//neuron_concentrations=C^T * Q for each neuron (inner product for each neuron = diagonal of matrix multiplication)
	for (int n = 0; n < nnz; n++) {
		neuron_concentrations[n] = 0.0; //reset 
		for (int j = 0; j < 8; j++) { //goes across rows of CT and columns of Q
			std::cout << "neuron concentration:" << std::endl;
			std::cout << neuron_concentrations[n]<<" += " << neuron_concentrations[n]<<" + " << CT[8 * n + j] <<" * " <<Q[nnz * j + n] << std::endl;
			neuron_concentrations[n] = neuron_concentrations[n] + CT[8*n+j] * Q[nnz*j+n]; //getting the diagonal elements of (C^T * Q) (matrix mult)
		}
	}

	std::cout << "Neuron Concentration: " <<neuron_concentrations[0] << std::endl;
	return neuron_concentrations;
}


//paracrine::paracrine(int grid_size, int nnz, const thrust::device_vector<float> neuron_x, const thrust::device_vector<float> neuron_y,\
//			const thrust::device_vector<float> neuron_z,const thrust::device_vector<float> grid_IC, const thrust::device_vector<float> neuron_IC) { //initialize paracrine class with this constructor
//			//call Initialization function
//			initialize(grid_size, nnz, neuron_x, neuron_y, neuron_z, grid_IC, neuron_IC);
//			std::cout << "Paracrine Initialized" << std::endl;
//};

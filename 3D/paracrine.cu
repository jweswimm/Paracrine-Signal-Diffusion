//Author: Joe Wimmergren 2022
#include "paracrine.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include "cuda_runtime.h""
#include "device_launch_parameters.h"

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
	thrust::host_vector<Float> del_x(nnz);
	thrust::host_vector<Float> del_y(nnz);
	thrust::host_vector<Float> del_z(nnz);


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

__global__ void gpu_interpolate(int nnz, int grid_size, Float* CT, Float* grid, Float* neuron_concentrations, Float* Q,
	int* x0, int* x1, int* y0, int* y1, int* z0, int* z1)
	//https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
{ //can't be member of class paracrine?
	//EACH NEURON GETS A THREAD

	int grid_height = grid_size;
	int grid_depth = grid_size;
	int rowsize = 8;







	for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nnz; row += gridDim.x * blockDim.x) //neuron loop
	{
		//Float* CT_ptr = CT_ptr + idx; //move the pointer depending on the neuron
		//Float CT = *CT_ptr; //get values of CT

		//Float* grid_ptr = grid_ptr + idx;
		//Float grid = *grid_ptr;

		//Float* x0_ptr = x0_ptr + idx;
		//Float x0 = *x0_ptr;

		//Float* x1_ptr = x1_ptr + idx;
		//Float x1 = *x1_ptr;

		//Float* y0_ptr = y0_ptr + idx;
		//Float y0 = *y0_ptr;

		//Float* y1_ptr = y1_ptr + idx;
		//Float y1 = *y1_ptr;

		//Float* z0_ptr = z0_ptr + idx;
		//Float z0 = *z0_ptr;

		//Float* z1_ptr = z1_ptr + idx;
		//Float z1 = *z1_ptr;

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
		//precalculate the indices !!!!!


		neuron_concentrations[row] = 0.0; //reset 
		for (int j = 0; j < 8; j++) { //goes across rows of CT and columns of Q
//			std::cout << "neuron concentration:" << std::endl;
//			std::cout << neuron_concentrations[n]<<" += " << neuron_concentrations[n]<<" + " << CT[8 * n + j] <<" * " <<Q[nnz * j + n] << std::endl;
			neuron_concentrations[row] = neuron_concentrations[row] + CT[8 * row + j] * Q[nnz * j + row]; //getting the diagonal elements of (C^T * Q) (matrix mult)
//		if (threadIdx.x == 0){
			//printf("%.6f \n",neuron_concentrations[row]);
//		}
		}

	}
}



thrust::device_vector<Float> paracrine::interpolate(int nnz, int grid_size, thrust::host_vector<Float> grid) { //return pointer to array of concentrations at neuron locations
//	thrust::host_vector<Float> neuron_concentrations(nnz);

	int grid_width = grid_size; 
	int grid_height = grid_size; 
	int grid_depth = grid_size; 


	int rowsize = 8;


	//assuming grid is already flattened (was grid_size x grid_size x grid_size, now is grid_size^3 x 1)
	//So grid[x0, y0, z0]=grid[grid_height*grid_depth*x0 + grid_depth*y0 + z0]

	//Build coefficient matrix C
	// and in the process, transpose it
	//initialize to 0
	thrust::host_vector<Float> CT(nnz*8,0.0);

	//------------------------CPU INTERPOLATION---------------------------------------------------------------------------------------
	//There will be c values for each element, e.g nnz many c0 values
	//Note: this must be computed every time the interpolation takes place, as the concentration
	//at the grid points will be changing(i.e. grid itself will be changing)
//		for (int row = 0; row < nnz; row++) {
////			std::cout <<"testvalue: "<< rowsize * row + column << std::endl;
//				CT[rowsize * row + 0] = grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]
//
//				CT[rowsize * row + 1] = grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]]  //grid[x1, y0, z0]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]
//
//				CT[rowsize * row + 2] = grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]]  //grid[x0, y1, z0]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]
//
//				CT[rowsize * row + 3] = grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]]  //grid[x0, y0, z1]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]
//
//				CT[rowsize * row + 4] = grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z0[row]] //grid[x1,y1,z0]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]]  //grid[x0,y1,z0]
//					- grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]]  //grid[x1,y0,z0]
//					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]
//
//				CT[rowsize * row + 5] = grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z1[row]] //grid[x0,y1,z1]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]]  //grid[x0,y0,z1]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]]  //grid[x0,y1,z0]
//					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]
//
//				CT[rowsize * row + 6] = grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z1[row]] //grid[x1,y0,z1]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]]  //grid[x0,y0,z1]
//					- grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]]  //grid[x1,y0,z0]
//					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]
//
//				CT[rowsize * row + 7] = grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z1[row]] //grid[x1,y1,z1]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z1[row]]  //grid[x0, y1, z1]
//					- grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z1[row]]  //grid[x1, y0, z1]
//					- grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z0[row]]  //grid[x1, y1, z0]
//					+ grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]]  //grid[x1, y0, z0]
//					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]]  //grid[x0, y0, z1]
//					+ grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]]  //grid[x0, y1, z0]
//					- grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]];  //grid[x0, y0, z0]
//			
//
//
//		}
//
//
//	//Now that we have CT created, we have to
//	//neuron_concentrations=C^T * Q for each neuron (inner product for each neuron = diagonal of matrix multiplication)
//	for (int n = 0; n < nnz; n++) {
//		neuron_concentrations[n] = 0.0; //reset 
//		for (int j = 0; j < 8; j++) { //goes across rows of CT and columns of Q
////			std::cout << "neuron concentration:" << std::endl;
////			std::cout << neuron_concentrations[n]<<" += " << neuron_concentrations[n]<<" + " << CT[8 * n + j] <<" * " <<Q[nnz * j + n] << std::endl;
//			neuron_concentrations[n] = neuron_concentrations[n] + CT[8*n+j] * Q[nnz*j+n]; //getting the diagonal elements of (C^T * Q) (matrix mult)
//		}
//	}
//-------------------------------------END OF CPU INTERPOLATION--------------------------------------------------------------------------

//	std::cout << "Neuron Concentration: " <<neuron_concentrations[0] << std::endl;

//-------------------------------------GPU INTERPOLATION----------------------------------------------------------------------------------
	// get vectors on device
	//allocate memory on device for d_CT
	thrust::device_vector<Float> d_CT = CT;
//	Float* d_CT;
//	Float* CT_ptr = thrust::raw_pointer_cast(CT.data());
//	cudaMalloc(&d_CT, 8 * nnz * sizeof(Float));
//	cudaMemcpy(d_CT, CT_ptr, 8*nnz*sizeof(Float), cudaMemcpyHostToDevice);


	thrust::device_vector<Float> d_grid = grid;
//	Float* d_grid;
//	Float* grid_ptr = thrust::raw_pointer_cast(grid.data());
//	cudaMalloc(&d_grid, 8 * nnz * sizeof(Float));
//	cudaMemcpy(d_grid, grid_ptr, 8*nnz*sizeof(Float), cudaMemcpyHostToDevice);

	thrust::device_vector<Float> d_neuron_concentrations(nnz);
	//Float* d_neuron_concentrations;
	//Float* neuron_concentrations_ptr = thrust::raw_pointer_cast(neuron_concentrations.data());
	//cudaMalloc(&d_neuron_concentrations, 8 * nnz * sizeof(Float));
	//cudaMemcpy(d_neuron_concentrations, neuron_concentrations_ptr, 8*nnz*sizeof(Float), cudaMemcpyHostToDevice);

	thrust::device_vector<Float> d_Q = Q;
	//Float* d_Q;
	//Float* Q_ptr = thrust::raw_pointer_cast(Q.data());
	//cudaMalloc(&d_Q, 8 * nnz * sizeof(Float));
	//cudaMemcpy(d_Q, Q_ptr, 8*nnz*sizeof(Float), cudaMemcpyHostToDevice);

	//Calculate blocksize and gridsize
	int block_size = 1024; //schedule a block full of number of neurons
	int cuda_grid_size = nnz / block_size + 1;


	//create double pointer to send into gpu_interpolate (i.e. by reference in the kernel)


	gpu_interpolate <<< cuda_grid_size, block_size >>> (nnz, grid_size, thrust::raw_pointer_cast(d_CT.data()), thrust::raw_pointer_cast(d_grid.data()),
		thrust::raw_pointer_cast(d_neuron_concentrations.data()), thrust::raw_pointer_cast(d_Q.data()), thrust::raw_pointer_cast(x0.data()), 
		thrust::raw_pointer_cast(x1.data()), thrust::raw_pointer_cast(y0.data()), thrust::raw_pointer_cast(y1.data()),
		thrust::raw_pointer_cast(z0.data()), thrust::raw_pointer_cast(z1.data()));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
//	std::cout << d_neuron_concentrations[0] << "TESTING" << std::endl;

	return d_neuron_concentrations;
}




	

__global__ void gpu_spread(int nnz, int grid_size, Float* grid, Float* P, 
	int* x0, int* x1, int* y0, int* y1, int* z0, int* z1){

	int grid_height = grid_size;
	int grid_depth = grid_size;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nnz; i += gridDim.x * blockDim.x){ //neuron loop
		grid[grid_height * grid_depth * x0[i] + grid_depth * y0[i] + z0[i]] += P[nnz * 0 + i];
		grid[grid_height * grid_depth * x0[i] + grid_depth * y0[i] + z1[i]] += P[nnz * 1 + i];
		grid[grid_height * grid_depth * x0[i] + grid_depth * y1[i] + z0[i]] += P[nnz * 2 + i];
		grid[grid_height * grid_depth * x0[i] + grid_depth * y1[i] + z1[i]] += P[nnz * 3 + i];
		grid[grid_height * grid_depth * x1[i] + grid_depth * y0[i] + z0[i]] += P[nnz * 4 + i];
		grid[grid_height * grid_depth * x1[i] + grid_depth * y0[i] + z1[i]] += P[nnz * 5 + i];
		grid[grid_height * grid_depth * x1[i] + grid_depth * y1[i] + z0[i]] += P[nnz * 6 + i];
		grid[grid_height * grid_depth * x1[i] + grid_depth * y1[i] + z1[i]] += P[nnz * 7 + i];
	}
}




//spread function
thrust::device_vector<Float> paracrine::spread(thrust::device_vector<Float> grid,thrust::device_vector<Float> neuron_concentrations ) {
	std::cout << "Spreading Started" << std::endl;
	
	//calculate neurotransmitter generation (thrust vector of size nnz)
	Float generation_constant = 1; //for now just have as a constant



	//---------------------------CPU SPREADING--------------------------------------------------------------------------------------
	//int grid_width = grid_size;
	//int grid_height = grid_size;
	//int grid_depth = grid_size;

	////	using namespace thrust::placeholders;
	//thrust::host_vector <Float> generation(nnz*8, generation_constant);
	////thrust::transform(neuron_concentrations.begin(), neuron_concentrations.end(), generation.begin(), generation_constant * _1); //_1 is a placeholder

	//thrust::host_vector <Float> P(nnz*8,0.0);


	//thrust::transform(generation.begin(), generation.end(), weighted_spread.begin(), P.begin(), thrust::multiplies<Float>()); //P=generation * weighted_spread element by element

	//for (int i=0; i<nnz; i++){ //weighted_spread[rowsize * 0 + column]
	//	grid[grid_height * grid_depth * x0[i] + grid_depth * y0[i] + z0[i]] += P[nnz*0 + i];
	//	grid[grid_height * grid_depth * x0[i] + grid_depth * y0[i] + z1[i]] += P[nnz*1 + i];
	//	grid[grid_height * grid_depth * x0[i] + grid_depth * y1[i] + z0[i]] += P[nnz*2 + i];
	//	grid[grid_height * grid_depth * x0[i] + grid_depth * y1[i] + z1[i]] += P[nnz*3 + i];
	//	grid[grid_height * grid_depth * x1[i] + grid_depth * y0[i] + z0[i]] += P[nnz*4 + i];
	//	grid[grid_height * grid_depth * x1[i] + grid_depth * y0[i] + z1[i]] += P[nnz*5 + i];
	//	grid[grid_height * grid_depth * x1[i] + grid_depth * y1[i] + z0[i]] += P[nnz*6 + i];
	//	grid[grid_height * grid_depth * x1[i] + grid_depth * y1[i] + z1[i]] += P[nnz*7 + i];
	//}
	//----------------------------END CPU SPREADING----------------------------------------------------------------------------------




	//-----------------------------GPU SPREADING--------------------------------------------------------------------------------------

	//Send weighted_spread (found in initialization) to gpu
	thrust::device_vector<Float> d_weighted_spread = weighted_spread;

	//Create thrust vector P
	thrust::device_vector<Float> d_P(nnz*8);

	//Create generation vector
	thrust::device_vector <Float> d_generation(nnz*8, generation_constant);


	//Create P vector (done on GPU still with thrust library)
	thrust::transform(d_generation.begin(), d_generation.end(), d_weighted_spread.begin(), d_P.begin(), thrust::multiplies<Float>()); //P=generation * weighted_spread element by element
	
	//Calculate blocksize and gridsize
	int block_size = 1024; //schedule a block full of number of neurons
	int cuda_grid_size = nnz / block_size + 1;

	gpu_spread <<< cuda_grid_size, block_size >>> (nnz, grid_size, thrust::raw_pointer_cast(grid.data()),
		thrust::raw_pointer_cast(d_P.data()), thrust::raw_pointer_cast(x0.data()), 
		thrust::raw_pointer_cast(x1.data()), thrust::raw_pointer_cast(y0.data()), thrust::raw_pointer_cast(y1.data()),
		thrust::raw_pointer_cast(z0.data()), thrust::raw_pointer_cast(z1.data()));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());


	return grid;
}


//3d Convolution kernel
__global__ void gpu_convolve(Float* image, Float* mask, float* result, int image_size, int mask_size ) {

}






thrust::device_vector < Float> paracrine::convolve(thrust::device_vector<Float> grid, thrust::device_vector<Float> mask) {
	std::cout << "Convolution started" << std::endl;

	//Mask is 3x3x3 (in our case, 27-point discrete laplacian stencil)
	//Mask dimensions are initialized in paracrine class


	//Pad the image with 0's so that we retain the same size after convolution
	int image_size = grid_size + 2;
	//initialize as 0's first
	thrust::device_vector<Float> image(image_size * image_size * image_size, 0.0); //same as grid but with 0's on boundaries in 3d

	//Copy values over from grid to center of image
	for (int i = 1; i < grid_size+1; i++) {
		for (int j = 1; j < grid_size+1; j++) {
			for (int k = 1; k < grid_size+1; k++) { //i,j,k are between 0 and grid_size for grid
				image[image_size * image_size * i + image_size * j + k] = grid[grid_size * grid_size * (i-1) + grid_size * (j-1) + (k-1)] ;
			}
		}
	}

	//We now have image. Image is just grid (from argument to this function) with 0s on the boundaries.
	//Now when we convolve, we won't lose any dimensions (our mask is 3x3x3 so we only needed 1 extra space in each dimension)
	
	//Call gpu_convolve function to do 3d convolution with mask and image




	//Create new 3d thrust vector that will eventually be output
	thrust::device_vector<Float> result(grid_size * grid_size * grid_size);

	
//	return result; //actual return
	return image;
}

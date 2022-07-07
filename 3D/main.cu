#include "paracrine.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>

void innerproducttest() {
	//Get all variables set so we can create paracrine object
	int grid_size = 2;
	int nnz = 1;
	Float dx = 1.0;
	Float dt = 1.0;
	Float diffusion = 1.0;
	Float decay = 1.0;
	thrust::host_vector<Float> neuron_locations_x(nnz);
	thrust::host_vector<Float> neuron_locations_y(nnz);
	thrust::host_vector<Float> neuron_locations_z(nnz);
	thrust::host_vector<Float> neuron_IC(nnz);
	thrust::host_vector<Float> grid_IC(grid_size*grid_size*grid_size);

	//initialize thrust vector
	//Set neuron locations very simple first (assume 0.5 in x,y,z directions)
	for (int i = 0; i < nnz; i++){
	neuron_locations_x[i] = 0.5;
	neuron_locations_y[i] = 0.5;
	neuron_locations_z[i] = 0.5;
	}

	//Set initial conditions on grid
	for (int i = 0; i < grid_size * grid_size * grid_size; i++)
		grid_IC[i] = 1;

	//initialize object
	paracrine IPtest(grid_size, nnz, dx, dt, 
		diffusion, decay, neuron_locations_x, 
		neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);

	//Now we want to test inner product
	//So create two vectors
	int size = 4;
	thrust::device_vector<Float> v1(size, 1.0);
	thrust::device_vector<Float> v2(size, 1.0);
	for (int i = 0; i < size; i++) {
		std::cout << "v1[" << i << "]=" << v1[i] << std::endl;
		std::cout << "v2[" << i << "]=" << v2[i] << std::endl;
		std::cout << std::endl;
	}
	std::cout<<"Inner product of v1 and v2 is "<<IPtest.inner_product(v1, v2);
}



//Conjugate Gradient Test
void CGtest() {
	//Get all variables set so we can create paracrine object
	int grid_size = 4;
	int nnz = 1;
	Float dx = 1.0;
	Float dt = 1.0;
	Float diffusion = 1.0;
	Float decay = 1.0;
	thrust::host_vector<Float> neuron_locations_x(nnz);
	thrust::host_vector<Float> neuron_locations_y(nnz);
	thrust::host_vector<Float> neuron_locations_z(nnz);
	thrust::host_vector<Float> neuron_IC(nnz);
	thrust::host_vector<Float> grid_IC(grid_size*grid_size*grid_size);

	//initialize thrust vector
	//Set neuron locations very simple first (assume 0.5 in x,y,z directions)
	for (int i = 0; i < nnz; i++){
	neuron_locations_x[i] = 0.5;
	neuron_locations_y[i] = 0.5;
	neuron_locations_z[i] = 0.5;
	}

	//Set initial conditions on grid
	for (int i = 0; i < grid_size * grid_size * grid_size; i++)
		grid_IC[i] = rand();

	//initialize object
	paracrine CGtest(grid_size, nnz, dx, dt, 
		diffusion, decay, neuron_locations_x, 
		neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);

	//CG solves Ax=b
	thrust::device_vector<Float> A(27, 1.0);
	thrust::device_vector<Float> b(grid_size * grid_size * grid_size, 10);
	int max_iterations = 1000;
	Float error_tol = 0.000001;

	//Now the answer to Ax=b should be x such that mask_mult(x,A)=b;
	thrust::device_vector<Float> x(grid_size * grid_size * grid_size);

	thrust::device_vector<Float> laplacian_grid = CGtest.mask_mult(grid_IC, A);

	x = CGtest.CG(A, b, grid_IC, laplacian_grid, max_iterations, error_tol);

	//Now we need to test mask_mult(x,A)=b;
	thrust::device_vector<Float> b_test(grid_size * grid_size * grid_size);
	b_test = CGtest.mask_mult(x, A);
	for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
		std::cout << "real b=" << b[i] << "   b_test=" << b_test[i] << std::endl;
	}


}

//Mask Multiplication Test
void MaskMulttest() {

	//Get all variables set so we can create paracrine object
	int grid_size = 8;
	int nnz = 1;
	Float dx = 1.0;
	Float dt = 1.0;
	Float diffusion = 1.0;
	Float decay = 1.0;
	thrust::host_vector<Float> neuron_locations_x(nnz);
	thrust::host_vector<Float> neuron_locations_y(nnz);
	thrust::host_vector<Float> neuron_locations_z(nnz);
	thrust::host_vector<Float> neuron_IC(nnz);
	thrust::host_vector<Float> grid_IC(grid_size*grid_size*grid_size);

	//initialize thrust vector
	//Set neuron locations very simple first (assume 0.5 in x,y,z directions)
	for (int i = 0; i < nnz; i++){
	neuron_locations_x[i] = 0.5;
	neuron_locations_y[i] = 0.5;
	neuron_locations_z[i] = 0.5;
	}

	//Set initial conditions on grid
	for (int i = 0; i < grid_size * grid_size * grid_size; i++)
		grid_IC[i] = rand();

	//initialize object
	paracrine MaskMulttest(grid_size, nnz, dx, dt, 
		diffusion, decay, neuron_locations_x, 
		neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);

	//For this test, given A (3x3x3) and x (grid_size x grid_size x grid_size), we should be able to compute Ax=b
	//This is done by sliding A across x and calculating the sum of the element wise multiplications
	thrust::device_vector<Float> A(27, 1.0);
	thrust::device_vector<Float> x(grid_size*grid_size*grid_size, 1.0);
	thrust::device_vector<Float> b(grid_size*grid_size*grid_size);
	b=MaskMulttest.mask_mult(x, A);

	//A is a 3x3x3 mask of 1's
	//x is grid_size x grid_size x grid_size of 1's
	//We should see when the mask isn't touching the boundary of x (i.e. when i,j, or k are not 0 or grid_size)
	//we expect the value of b to be 27.
	//If there is 1 side of the mask touching the boundary (i.e. when i,j, or k are equal to 0 or grid_size)
	//we expect the value of b to be 18.
	//If there are 2 sides of the mask touching the boundary (i.e. when i and j, or j and k, or k and i, are 0 or grid_size)
	//we expect the value of b to be 12.
	//If there are 3 sides of the mask touching the boundary
	//we expect the value of b to be 8.
	for (int i = 0; i < grid_size ; i++) {
		for (int j = 0; j < grid_size; j++) {
			for (int k = 0; k < grid_size; k++) {
				std::cout << "b[" <<i<<","<<j<<","<<k<< "]=" << b[grid_size*grid_size*i + grid_size * j +k] << std::endl;
			}
		}
	}
}


//Interpolation Test
void interpolationtest() {

}


//https://www.sie.es/wp-content/uploads/2015/12/Intro-to-Thrust-Parallel-Algorithms-Library.pdf
int main() {
//	innerproducttest();
//	CGtest();
//	MaskMulttest();


	//to do: 
	//convert 
	//create thrust host_vector
//	const int nnz = 1024;
//	const int grid_size = 32; //grid(grid_size, grid_size, grid_size), assuming cube uniform grid
//	thrust::host_vector<Float> neuron_locations_x(nnz+1);
//	thrust::host_vector<Float> neuron_locations_y(nnz+1);
//	thrust::host_vector<Float> neuron_locations_z(nnz+1);
//	thrust::host_vector<Float> neuron_IC(nnz+1);
//	thrust::host_vector<Float> grid_IC(grid_size*grid_size*grid_size+1);

	//initialize thrust vector
	//Set neuron locations very simple first (assume 0.5 in x,y,z directions)
	//will eventually need 3 1D vectors
//	for (int i = 0; i < nnz; i++){
//	neuron_locations_x[i] = 0.5;
//	neuron_locations_y[i] = 0.5;
//	neuron_locations_z[i] = 0.5;


//	neuron_locations_x[i] = rand() % 31;
//	neuron_locations_y[i] = rand() % 31;
//	neuron_locations_z[i] = rand() % 31;
//}

	//Set initial conditions on grid
//	for (int i = 0; i < grid_size * grid_size * grid_size; i++)
//		//grid_IC[i] = rand() % 100;
//		grid_IC[i] = 1;

	//for (int x = 0; x < grid_size; x++) {
	//	for (int y = 0; y < grid_size; y++) {
	//		for (int z = 0; z < grid_size; z++) {
	//			grid_IC[grid_size * grid_size * x + grid_size * y + z]=0;	//grid[grid_height * grid_depth * x + grid_depth * y + z] = grid[x,y,z]
//
//			}
//		}
//	}

	//set some test gridpoints
	//grid_IC[grid_size * grid_size * 0 + grid_size *0 + 0]=1;	//grid[0,0,0]=0
	//grid_IC[grid_size * grid_size * 1 + grid_size *0 + 0]=1;	//grid[1,0,0]=0
	//grid_IC[grid_size * grid_size * 0 + grid_size *1 + 0]=1;	//grid[0,1,0]=0
	//grid_IC[grid_size * grid_size * 1 + grid_size *1 + 0]=1;	//grid[1,1,0]=0
	//grid_IC[grid_size * grid_size * 0 + grid_size *0 + 1]=1;	//grid[0,0,1]=1
	//grid_IC[grid_size * grid_size * 1 + grid_size *0 + 1]=1;	//grid[1,0,1]=1
	//grid_IC[grid_size * grid_size * 0 + grid_size *1 + 1]=1;	//grid[0,1,1]=1
	//grid_IC[grid_size * grid_size * 1 + grid_size *1 + 1]=1;	//grid[1,1,1]=1
	//grid[grid_height * grid_depth * x + grid_depth * y + z] = grid[x,y,z]


	//copy values to host
//	thrust::host_vector<Float> d_neuron_locations = neuron_locations;
//	thrust::host_vector<Float> d_grid_values = grid_values;
//	Float dx = 1; 
//	Float dt = 1;
//	Float diffusion = 1;
//	Float decay = 1;
	
	
	//create paracrine object
//	paracrine ptest(grid_size, nnz, dx, dt, diffusion, decay, neuron_locations_x, neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);

	//initialize paracrine 
//	ptest.initialize(); //get Q and weighted_spread vectors to prepare for interpolation, spreading, and diffusion
//	ptest.diffusion_stepper(grid_IC, 1, 1, 1, 1);


	//We now have the initial conditions on the grid, so we have to interpolate to the neuron to determine how much neurotransmitter concentration
	//is at the neuron location
//	thrust::device_vector<Float> paracrine_at_neuron(nnz+1);

	//-----------------------------//
//	thrust::device_vector<Float> d_neuron_conc(nnz);
//	d_neuron_conc=ptest.interpolate(nnz,grid_size,grid_IC);

	//grid=ptest.spread(grid_IC, paracrine_at_neuron);

	//copy values from device to host to display them
//	cudaMemcpy(thrust::raw_pointer_cast(paracrine_neuron_conc.data()), paracrine_at_neuron_ptr, nnz * sizeof(Float), cudaMemcpyDeviceToHost);
//	cudaThreadSynchronize();



//	std::cout << "neuron_concentration = " << d_neuron_conc[0] << std::endl;
	//std::cout << "Testing interp and spread" << std::endl;
	//std::cout << grid_IC[0] << std::endl;
	//std::cout << grid[0] << std::endl;



//	std::cout << "Spreading Test:" << std::endl;;
//	thrust::device_vector<Float> grid(grid_size * grid_size * grid_size+1);
//	grid = ptest.spread(grid_IC, d_neuron_conc);
//	std::cout << grid[0] << std::endl;






	


	//test convolve
	//std::cout << "Convolve tester: "<<grid[grid_size * grid_size * 20 + grid_size * 20 + 20] << std::endl;
//	thrust::device_vector<Float> grid(grid_size*grid_size*grid_size);
//	grid = ptest.stencil_mult(grid_IC);
	//std::cout << "New  grid value: " << grid[grid_size * grid_size * 20 + grid_size * 20 + 20] << std::endl;

	
	//Test the padding
	//std::cout << "Image has values: " << std::endl;
	//for (int i=1; i<grid_size+1; i++){
	//	for (int j = 1; j < grid_size + 1; j++) {
	//			std::cout<<image[(grid_size + 2) * i * (grid_size+2) + (grid_size + 2) * (grid_size+1) + j] << std::endl;
	//	}
	//}




}

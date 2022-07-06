#include "paracrine.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>


int main() {
	//to do: 
	//convert 
	//create thrust host_vector
	const int nnz = 1024;
	const int grid_size = 32; //grid(grid_size, grid_size, grid_size), assuming cube uniform grid
	thrust::host_vector<Float> neuron_locations_x(nnz+1);
	thrust::host_vector<Float> neuron_locations_y(nnz+1);
	thrust::host_vector<Float> neuron_locations_z(nnz+1);
	thrust::host_vector<Float> neuron_IC(nnz+1);
	thrust::host_vector<Float> grid_IC(grid_size*grid_size*grid_size+1);

	//initialize thrust vector
	//Set neuron locations very simple first (assume 0.5 in x,y,z directions)
	//will eventually need 3 1D vectors
	for (int i = 0; i < nnz; i++){
	neuron_locations_x[i] = 0.5;
	neuron_locations_y[i] = 0.5;
	neuron_locations_z[i] = 0.5;


//	neuron_locations_x[i] = rand() % 31;
//	neuron_locations_y[i] = rand() % 31;
//	neuron_locations_z[i] = rand() % 31;
	}

	//Set initial conditions on grid
	for (int i = 0; i < grid_size * grid_size * grid_size; i++)
		//grid_IC[i] = rand() % 100;
		grid_IC[i] = 1;

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
	
	//create paracrine object
	paracrine ptest(grid_size, nnz, neuron_locations_x, neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);

	//initialize paracrine 
	ptest.initialize(); //get Q and weighted_spread vectors to prepare for interpolation, spreading, and diffusion
	ptest.diffusion_stepper(grid_IC);


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

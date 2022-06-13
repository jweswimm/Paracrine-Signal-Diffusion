#include "paracrine.cuh"
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main() {
	//create thrust device_vector
	const int nnz = 1;
	const int grid_size = 2; //grid(grid_size, grid_size, grid_size), assuming cube uniform grid
	thrust::device_vector<float> neuron_locations(nnz);
	thrust::device_vector<float> neuron_IC(nnz);
	thrust::device_vector<float> grid_IC(grid_size*grid_size*grid_size);

	//initialize thrust vector
	//Set neuron locations very simple first (assume 0.5 in x,y,z directions)
	//will eventually need 3 1D vectors
	for (int i = 0; i < nnz; i++)
		neuron_locations[i] = i+0.5;

	//Set initial conditions on grid
	//for (int i = 0; i < grid_size * grid_size * grid_size; i++)
	//	grid_IC[i] = i;

	//for (int x = 0; x < grid_size; x++) {
	//	for (int y = 0; y < grid_size; y++) {
	//		for (int z = 0; z < grid_size; z++) {
	//			grid_IC[grid_size * grid_size * x + grid_size * y + z]=0;	//grid[grid_height * grid_depth * x + grid_depth * y + z] = grid[x,y,z]
//
//			}
//		}
//	}

	//set some test gridpoints
	grid_IC[grid_size * grid_size * 0 + grid_size *0 + 0]=1;	//grid[0,0,0]=0
	grid_IC[grid_size * grid_size * 1 + grid_size *0 + 0]=1;	//grid[1,0,0]=0
	grid_IC[grid_size * grid_size * 0 + grid_size *1 + 0]=1;	//grid[0,1,0]=0
	grid_IC[grid_size * grid_size * 1 + grid_size *1 + 0]=1;	//grid[1,1,0]=0
	grid_IC[grid_size * grid_size * 0 + grid_size *0 + 1]=4;	//grid[0,0,1]=1
	grid_IC[grid_size * grid_size * 1 + grid_size *0 + 1]=4;	//grid[1,0,1]=1
	grid_IC[grid_size * grid_size * 0 + grid_size *1 + 1]=4;	//grid[0,1,1]=1
	grid_IC[grid_size * grid_size * 1 + grid_size *1 + 1]=4;	//grid[1,1,1]=1
	//grid[grid_height * grid_depth * x + grid_depth * y + z] = grid[x,y,z]


	//copy values to device
//	thrust::device_vector<float> d_neuron_locations = neuron_locations;
//	thrust::device_vector<float> d_grid_values = grid_values;
	
	//create paracrine object
	paracrine ptest(grid_size, nnz, neuron_locations, neuron_locations, neuron_locations, grid_IC, neuron_IC);

	//initialize paracrine 
	ptest.initialize(); //get Q and weighted_spread vectors to prepare for interpolation, spreading, and diffusion


	//We now have the initial conditions on the grid, so we have to interpolate to the neuron to determine how much neurotransmitter concentration
	//is at the neuron location
	thrust::device_vector<float> paracrine_at_neuron(nnz);
	paracrine_at_neuron=ptest.interpolate(nnz,grid_size,grid_IC);

	//next do spreading




}
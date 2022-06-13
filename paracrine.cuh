//Header File for Paracrine Part of SCN
#pragma once
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//Create Paracrine Class
class paracrine {
	public:
		const int grid_size; //size of 1 dimension of the grid (assuming cube grid)
		const int nnz; //number of neurons
		thrust::device_vector<float> neuron_x; //locations of neurons in x direction
		thrust::device_vector<float> neuron_y; //locations of neurons in y direction
		thrust::device_vector<float> neuron_z; //locations of neurons in z direction
		thrust::device_vector<float> grid_IC; //IC for each gridpoint (size grid_size**3)
		thrust::device_vector<float> neuron_IC; //IC for each neuron (size nnz)

		thrust::device_vector<float> Q; //used in trilinear interpolation
		thrust::device_vector<float> weighted_spread; //used in trilinear interpolation


		thrust::device_vector<float> ones; //vector of ones (size nnz) used in spreading


		//save Q and weighted_spread functions here too
		//save x0, x1, y0, y1, z0, z1 as well
	//declare closest gridpoint vectors
		thrust::device_vector<int> x0;
		thrust::device_vector<int> x1;
		thrust::device_vector<int> y0;
		thrust::device_vector<int> y1;
		thrust::device_vector<int> z0;
		thrust::device_vector<int> z1;


		void initialize();
		thrust::device_vector<float> paracrine::interpolate(int nnz, int grid_size, thrust::device_vector<float> grid);
		//Paracrine Constructor
		paracrine(const int grid_size, const int nnz, const thrust::device_vector<float> neuron_x, const thrust::device_vector<float> neuron_y, \
				  const thrust::device_vector<float> neuron_z, const thrust::device_vector<float> grid_IC, const thrust::device_vector<float> neuron_IC)\
			      : grid_size(grid_size), nnz(nnz), neuron_x(neuron_x), neuron_y(neuron_y), neuron_z(neuron_z), grid_IC(grid_IC), neuron_IC(neuron_IC)
		{//gather parameters by this constructor
			std::cout << "Paracrine Parameters Saved" << std::endl;
		};

};



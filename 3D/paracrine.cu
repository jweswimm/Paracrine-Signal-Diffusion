#include "paracrine.cuh"

using namespace thrust::placeholders;

//Paracrine Initialization Function
void paracrine::initialize() {
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
  // BE CAREFUL ABOUT CEIL FUNCTION AND OVERFLOAT
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

  //Now calculate distance vectors
  thrust::host_vector < Float > del_x(nnz);
  thrust::host_vector < Float > del_y(nnz);
  thrust::host_vector < Float > del_z(nnz);

  //we can create functor to apply the same operation across all of the elements of the thrust vector
  //see struct del_operator in paracrine.cuh
  //https://www.bu.edu/pasi/files/2011/07/Lecture6.pdf page 17 for an example of building custom thrust operations
  //for now, implement naiive way with for loop

  for (int i = 0; i < nnz; i++) {
    del_x[i] = ((neuron_x[i] - x0[i]) / (x1[i] - x0[i]));
    del_y[i] = ((neuron_y[i] - y0[i]) / (y1[i] - y0[i]));
    del_z[i] = ((neuron_z[i] - z0[i]) / (z1[i] - z0[i]));
  }

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

  //Initialize weighted spread to 0.0 so we can use unique indices
  for (int i = 0; i < nnz * 8; i++)
    weighted_spread.push_back(0.0);

  //Now create weighted_spread (used in spreading)
  //note that this is the same size as Q (8 x nnz)
  Float dx = 1.0;
  for (int column = 0; column < nnz; column++) { //[columnsize * column + row]
    weighted_spread[rowsize * 0 + column] = (ones[column] - del_x[column]) * (ones[column] - del_y[column]) * (ones[column] - del_z[column]);
    weighted_spread[rowsize * 1 + column] = (ones[column] - del_x[column]) * (ones[column] - del_y[column]) * (del_z[column]);
    weighted_spread[rowsize * 2 + column] = (ones[column] - del_x[column]) * (del_y[column]) * (ones[column] - del_z[column]);
    weighted_spread[rowsize * 3 + column] = (ones[column] - del_x[column]) * (del_y[column]) * (del_z[column]);
    weighted_spread[rowsize * 4 + column] = (del_x[column]) * (ones[column] - del_y[column]) * (ones[column] - del_z[column]);
    weighted_spread[rowsize * 5 + column] = (del_x[column]) * (ones[column] - del_y[column]) * (del_z[column]);
    weighted_spread[rowsize * 6 + column] = (del_x[column]) * (del_y[column]) * (ones[column] - del_z[column]);
    weighted_spread[rowsize * 7 + column] = (del_x[column]) * (del_y[column]) * (del_z[column]);
  }

  //Initialize Diffusion Part
  //Initialize Stencil, eye, A, B (used in diffusion)
  for (int i = 0; i < 27; i++) {
    stencil.push_back(0.0);
    eye.push_back(0.0);
    A.push_back(0.0);
    B.push_back(0.0);
  }
  int stencil_height = 3;
  int stencil_depth = 3;

  //The stencil is flattened using the scheme stencil[x,y,z] = stencil[stencil_height * stencil_depth * x + stencil_depth * y + z];
  //x=0 plane
  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 0 + 0] = 1 / (30 * dx * dx); //stencil[0,0,0]
  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 1 + 0] = 3 / (30 * dx * dx); //stencil[0,1,0]
  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 2 + 0] = 1 / (30 * dx * dx); //stencil[0,2,0]

  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 0 + 1] = 3 / (30 * dx * dx); //stencil[0,0,1]
  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 1 + 1] = 14 / (30 * dx * dx); //stencil[0,1,1]
  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 2 + 1] = 3 / (30 * dx * dx); //stencil[0,2,1]

  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 0 + 2] = 1 / (30 * dx * dx); //stencil[0,0,2]
  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 1 + 2] = 3 / (30 * dx * dx); //stencil[0,1,2]
  stencil[stencil_height * stencil_depth * 0 + stencil_depth * 2 + 2] = 1 / (30 * dx * dx); //stencil[0,2,2]

  //x=1 plane
  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 0 + 0] = 3 / (30 * dx * dx); //stencil[1,0,0]
  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 1 + 0] = 14 / (30 * dx * dx); //stencil[1,1,0]
  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 2 + 0] = 3 / (30 * dx * dx); //stencil[1,2,0]

  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 0 + 1] = 14 / (30 * dx * dx); //stencil[1,0,1]
  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 1 + 1] = -128 / (30 * dx * dx); //stencil[1,1,1]
  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 2 + 1] = 14 / (30 * dx * dx); //stencil[1,2,1]

  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 0 + 2] = 3 / (30 * dx * dx); //stencil[1,0,2]
  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 1 + 2] = 14 / (30 * dx * dx); //stencil[1,1,2]
  stencil[stencil_height * stencil_depth * 1 + stencil_depth * 2 + 2] = 3 / (30 * dx * dx); //stencil[1,2,2]

  //x=2 plane
  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 0 + 0] = 1 / (30 * dx * dx); //stencil[2,0,0]
  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 1 + 0] = 3 / (30 * dx * dx); //stencil[2,1,0]
  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 2 + 0] = 1 / (30 * dx * dx); //stencil[2,2,0]

  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 0 + 1] = 3 / (30 * dx * dx); //stencil[2,0,1]
  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 1 + 1] = 14 / (30 * dx * dx); //stencil[2,1,1]
  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 2 + 1] = 3 / (30 * dx * dx); //stencil[2,2,1]

  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 0 + 2] = 1 / (30 * dx * dx); //stencil[2,0,2]
  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 1 + 2] = 3 / (30 * dx * dx); //stencil[2,1,2]
  stencil[stencil_height * stencil_depth * 2 + stencil_depth * 2 + 2] = 1 / (30 * dx * dx); //stencil[2,2,2]

  //Create 3D identity matrix "eye"
  //x=0 plane
  eye[stencil_height * stencil_depth * 0 + stencil_depth * 0 + 0] = 1;
  eye[stencil_height * stencil_depth * 0 + stencil_depth * 1 + 1] = 1;
  eye[stencil_height * stencil_depth * 0 + stencil_depth * 2 + 2] = 1;
  //x=1 plane
  eye[stencil_height * stencil_depth * 1 + stencil_depth * 0 + 0] = 1;
  eye[stencil_height * stencil_depth * 1 + stencil_depth * 1 + 1] = 1;
  eye[stencil_height * stencil_depth * 1 + stencil_depth * 2 + 2] = 1;
  //x=2 plane
  eye[stencil_height * stencil_depth * 2 + stencil_depth * 0 + 0] = 1;
  eye[stencil_height * stencil_depth * 2 + stencil_depth * 1 + 1] = 1;
  eye[stencil_height * stencil_depth * 2 + stencil_depth * 2 + 2] = 1;

  //set diffusion/decay constant (make this class wide later)
  Float diffusion = 1;
  Float decay = 1;
  Float dt = 0.01;

  for (int i = 0; i < 27; i++) {
    A[i] = eye[i] + 0.5 * diffusion * stencil[i] * dt + 0.5 * dt * decay * eye[i];
    B[i] = eye[i] - 0.5 * diffusion * stencil[i] * dt - 0.5 * dt * decay * eye[i];
  }

}

__global__ void gpu_interpolate(int nnz, int grid_size, Float * CT, Float * grid, Float * neuron_concentrations, Float * Q,
  int * x0, int * x1, int * y0, int * y1, int * z0, int * z1)
//https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
{
  //Each neuron gets a thread 
  int grid_height = grid_size;
  int grid_depth = grid_size;
  int rowsize = 8;

  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nnz; row += gridDim.x * blockDim.x) //neuron loop
  {
    
    CT[rowsize * row + 0] = grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

    CT[rowsize * row + 1] = grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]] //grid[x1, y0, z0]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

    CT[rowsize * row + 2] = grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]] //grid[x0, y1, z0]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

    CT[rowsize * row + 3] = grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]] //grid[x0, y0, z1]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

    CT[rowsize * row + 4] = grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z0[row]] //grid[x1,y1,z0]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]] //grid[x0,y1,z0]
      -
      grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]] //grid[x1,y0,z0]
      +
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

    CT[rowsize * row + 5] = grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z1[row]] //grid[x0,y1,z1]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]] //grid[x0,y0,z1]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]] //grid[x0,y1,z0]
      +
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

    CT[rowsize * row + 6] = grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z1[row]] //grid[x1,y0,z1]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]] //grid[x0,y0,z1]
      -
      grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]] //grid[x1,y0,z0]
      +
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]

    CT[rowsize * row + 7] = grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z1[row]] //grid[x1,y1,z1]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z1[row]] //grid[x0, y1, z1]
      -
      grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z1[row]] //grid[x1, y0, z1]
      -
      grid[grid_height * grid_depth * x1[row] + grid_depth * y1[row] + z0[row]] //grid[x1, y1, z0]
      +
      grid[grid_height * grid_depth * x1[row] + grid_depth * y0[row] + z0[row]] //grid[x1, y0, z0]
      +
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z1[row]] //grid[x0, y0, z1]
      +
      grid[grid_height * grid_depth * x0[row] + grid_depth * y1[row] + z0[row]] //grid[x0, y1, z0]
      -
      grid[grid_height * grid_depth * x0[row] + grid_depth * y0[row] + z0[row]]; //grid[x0, y0, z0]
    //TODO: precalculate the indices

    neuron_concentrations[row] = 0.0; //reset 
    for (int j = 0; j < 8; j++) { //goes across rows of CT and columns of Q
      //neuron_concentrations[row] = neuron_concentrations[row] + CT[8 * row + j] * Q[nnz * j + row]; //getting the diagonal elements of (C^T * Q) (matrix mult)
      atomicAdd( & neuron_concentrations[row], CT[8 * row + j] * Q[nnz * j + row]); //getting the diagonal elements of (C^T * Q) (matrix mult)
	    
    }

  }
	
}

thrust::device_vector < Float > paracrine::interpolate(int nnz, int grid_size, thrust::device_vector < Float > grid) { //return  thrust array of concentrations at neuron locations
 
  std::cout << "Interpolation Started" << std::endl;

  int grid_width = grid_size;
  int grid_height = grid_size;
  int grid_depth = grid_size;

  int rowsize = 8;

  //assuming grid is already flattened (was grid_size x grid_size x grid_size, now is grid_size^3 x 1)
  //So grid[x0, y0, z0]=grid[grid_height*grid_depth*x0 + grid_depth*y0 + z0]

  //Build coefficient matrix C
  // and in the process, transpose it
  //initialize to 0
  thrust::host_vector < Float > CT(nnz * 8, 0.0);

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


  //-------------------------------------GPU INTERPOLATION----------------------------------------------------------------------------------
	
  //Create device vectors
  thrust::device_vector < Float > d_CT = CT;
  thrust::device_vector < Float > d_neuron_concentrations(nnz);
  thrust::device_vector < Float > d_Q = Q;
  
  //Calculate blocksize and gridsize
  int block_size = 1024; //schedule a block full of number of neurons
  int cuda_grid_size = nnz / block_size + 1;

  gpu_interpolate << < cuda_grid_size, block_size >>> (nnz, grid_size, thrust::raw_pointer_cast(d_CT.data()), thrust::raw_pointer_cast(grid.data()),
    thrust::raw_pointer_cast(d_neuron_concentrations.data()), thrust::raw_pointer_cast(d_Q.data()), thrust::raw_pointer_cast(x0.data()),
    thrust::raw_pointer_cast(x1.data()), thrust::raw_pointer_cast(y0.data()), thrust::raw_pointer_cast(y1.data()),
    thrust::raw_pointer_cast(z0.data()), thrust::raw_pointer_cast(z1.data()));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return d_neuron_concentrations;
}

__global__ void gpu_spread(int nnz, int grid_size, Float * grid, Float * P,
  int * x0, int * x1, int * y0, int * y1, int * z0, int * z1) {

  int grid_height = grid_size;
  int grid_depth = grid_size;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nnz; i += gridDim.x * blockDim.x) { //neuron loop
    //check that p is constructed correctly NEED ATOMIC ADD
    /*grid[grid_height * grid_depth * x0[i] + grid_depth * y0[i] + z0[i]] += P[nnz * 0 + i];
    grid[grid_height * grid_depth * x0[i] + grid_depth * y0[i] + z1[i]] += P[nnz * 1 + i];
    grid[grid_height * grid_depth * x0[i] + grid_depth * y1[i] + z0[i]] += P[nnz * 2 + i];
    grid[grid_height * grid_depth * x0[i] + grid_depth * y1[i] + z1[i]] += P[nnz * 3 + i];
    grid[grid_height * grid_depth * x1[i] + grid_depth * y0[i] + z0[i]] += P[nnz * 4 + i];
    grid[grid_height * grid_depth * x1[i] + grid_depth * y0[i] + z1[i]] += P[nnz * 5 + i];
    grid[grid_height * grid_depth * x1[i] + grid_depth * y1[i] + z0[i]] += P[nnz * 6 + i];
    grid[grid_height * grid_depth * x1[i] + grid_depth * y1[i] + z1[i]] += P[nnz * 7 + i];*/

    atomicAdd( & grid[grid_height * grid_depth * x0[i] + grid_depth * y0[i] + z0[i]], P[nnz * 0 + i]);
    atomicAdd( & grid[grid_height * grid_depth * x0[i] + grid_depth * y0[i] + z1[i]], P[nnz * 1 + i]);
    atomicAdd( & grid[grid_height * grid_depth * x0[i] + grid_depth * y1[i] + z0[i]], P[nnz * 2 + i]);
    atomicAdd( & grid[grid_height * grid_depth * x0[i] + grid_depth * y1[i] + z1[i]], P[nnz * 3 + i]);
    atomicAdd( & grid[grid_height * grid_depth * x1[i] + grid_depth * y0[i] + z0[i]], P[nnz * 4 + i]);
    atomicAdd( & grid[grid_height * grid_depth * x1[i] + grid_depth * y0[i] + z1[i]], P[nnz * 5 + i]);
    atomicAdd( & grid[grid_height * grid_depth * x1[i] + grid_depth * y1[i] + z0[i]], P[nnz * 6 + i]);
    atomicAdd( & grid[grid_height * grid_depth * x1[i] + grid_depth * y1[i] + z1[i]], P[nnz * 7 + i]);
  }

}

//spread function
thrust::device_vector < Float > paracrine::spread(Float generation_constant, 
	thrust::device_vector < Float > grid, thrust::device_vector < Float > neuron_concentrations) {
  std::cout << "Spreading Started" << std::endl;

  //calculate neurotransmitter generation (thrust vector of size nnz)
  //	Float generation_constant = 1; //for now just have as a constant

  //---------------------------CPU SPREADING--------------------------------------------------------------------------------------
  //int grid_width = grid_size;
  //int grid_height = grid_size;
  //int grid_depth = grid_size;

  ////	using namespace thrust::placeholders;
  //thrust::host_vector <Float> generation(nnz*8, generation_constant);
  ////thrust::transform(neuron_concentrations.begin(), neuron_concentrations.end(),
	//generation.begin(), generation_constant * _1); //_1 is a placeholder

  //thrust::host_vector <Float> P(nnz*8,0.0);

  //thrust::transform(generation.begin(), generation.end(), weighted_spread.begin(), 
	//P.begin(), thrust::multiplies<Float>()); //P=generation * weighted_spread element by element

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
  thrust::device_vector < Float > d_weighted_spread = weighted_spread;

  //Create thrust vector P
  thrust::device_vector < Float > d_P(nnz * 8);

  //Create generation vector
  thrust::device_vector < Float > d_generation(nnz * 8, generation_constant);

  //Create P vector (done on GPU still with thrust library)
  thrust::transform(d_generation.begin(), d_generation.end(), d_weighted_spread.begin(), d_P.begin(), thrust::multiplies < float > ()); //P=generation * weighted_spread element by element

  //Calculate blocksize and gridsize
  int block_size = 1024; //schedule a block full of threads
  int cuda_grid_size = nnz / block_size + 1;

  gpu_spread << < cuda_grid_size, block_size >>> (nnz, grid_size, thrust::raw_pointer_cast(grid.data()),
    thrust::raw_pointer_cast(d_P.data()), thrust::raw_pointer_cast(x0.data()),
    thrust::raw_pointer_cast(x1.data()), thrust::raw_pointer_cast(y0.data()), thrust::raw_pointer_cast(y1.data()),
    thrust::raw_pointer_cast(z0.data()), thrust::raw_pointer_cast(z1.data()));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return grid;
}

//3d Convolution kernel
__global__ void gpu_mask_mult(Float * image, Float * stencil, Float * result, int image_size, int mask_size, int grid_size) {

  //image is size ((grid_size+2) x (grid_size+2) x (grid_size+2))
  //stencil is size (3x3x3)
  //result will be (grid_size x grid_size x grid_size) thanks to the zero padding we did earlier

  //remember the flattening scheme
  //grid[grid_height * grid_depth * x + grid_depth * y + z] = grid[x,y,z]

  //Use the 3d architecture of the GPU to our advantage
  //nested loops to give result(i,j,k)
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < grid_size; i += gridDim.x * blockDim.x) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < grid_size; j += gridDim.y * blockDim.y) {
      for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < grid_size; k += gridDim.z * blockDim.z) {

        //Now multiply element wise and sum to get actual values
        //Float sum = 0.0;
        for (int ii = 0; ii < mask_size; ii++) {
          for (int jj = 0; jj < mask_size; jj++) {
            for (int kk = 0; kk < mask_size; kk++) {
              //sum += (image[image_size * image_size * (i+ii) + image_size * (j+jj) + (k+kk)])
              //	    * (stencil[mask_size * mask_size * ii + mask_size*jj + kk]);
              atomicAdd( & result[grid_size * grid_size * i + grid_size * j + k], (image[image_size * image_size * (i + ii) + image_size * (j + jj) + (k + kk)]) *
                (stencil[mask_size * mask_size * ii + mask_size * jj + kk]));
            }
          }
        }

        //result[grid_size * grid_size * i + grid_size * j + k] = sum;

      }
    }
  }

}

thrust::device_vector < Float > paracrine::mask_mult(thrust::device_vector < Float > grid, thrust::device_vector < Float > mask) {

  int mask_size = 3;

  //Mask is 3x3x3 (in our case, 27-point discrete laplacian stencil)
  //Mask dimensions are initialized in paracrine class

  //Pad the image with 0's so that we retain the same size after convolution
  int image_size = grid_size + 2;
  //initialize as 0's first
  thrust::device_vector < Float > image(image_size * image_size * image_size, 0.0); //same as grid but with 0's on boundaries in 3d

  //Copy values over from grid to center of image
  for (int i = 1; i < grid_size + 1; i++) {
    for (int j = 1; j < grid_size + 1; j++) {
      for (int k = 1; k < grid_size + 1; k++) { //i,j,k are between 0 and grid_size for grid
        image[image_size * image_size * i + image_size * j + k] = grid[grid_size * grid_size * (i - 1) + grid_size * (j - 1) + (k - 1)];
      }
    }
  }

  //We now have image. Image is just grid (from argument to this function) with 0s on the boundaries.
  //Now when we convolve, we won't lose any dimensions (our stencil is 3x3x3 so we only needed 1 extra space in each dimension)

  //Create new 3d thrust vector that will eventually be output
  thrust::device_vector < Float > result(grid_size * grid_size * grid_size, 0.0);

  //Call gpu_convolve function to do 3d convolution with mask and image

  //We want a thread for each element in result i.e. grid_size x grid_size x grid_size number of threads

  int threadsPerBlock_dimension = 8; //CANT HAVE MORE THAN 1024 THREADS PER BLOCK
  //allow for dynamically sized grid
  int gridWidth = ceil(Float(grid_size) / Float(threadsPerBlock_dimension));
  int gridHeight = ceil(Float(grid_size) / Float(threadsPerBlock_dimension));
  int gridDepth = ceil(Float(grid_size) / Float(threadsPerBlock_dimension));

  dim3 gridDim(gridWidth, gridHeight, gridDepth);
  dim3 blockDim(threadsPerBlock_dimension, threadsPerBlock_dimension, threadsPerBlock_dimension);

  gpu_mask_mult << < gridDim, blockDim >> > (thrust::raw_pointer_cast(image.data()), thrust::raw_pointer_cast(mask.data()),
    thrust::raw_pointer_cast(result.data()), image_size, mask_size, grid_size);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return result;
}

__global__ void gpu_initial_guess(int grid_size, Float diffusion, Float decay, Float dt, Float * grid, Float * laplacian_grid, Float * x) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < grid_size * grid_size * grid_size; i += gridDim.x * blockDim.x) {
    x[i] = (1 - dt * decay) * grid[i] + dt * diffusion * laplacian_grid[i];
  }
}

thrust::device_vector < Float > paracrine::initial_guess(thrust::device_vector < Float > grid, thrust::device_vector < Float > laplacian_grid) {
  thrust::device_vector < Float > x(grid_size * grid_size * grid_size);
  int block_size = 512; //schedule a block full of threads
  int cuda_grid_size = ceil((grid_size * grid_size * grid_size) / block_size);

  gpu_initial_guess << < cuda_grid_size, block_size >> > (grid_size, diffusion, decay, dt,
    thrust::raw_pointer_cast(grid.data()), thrust::raw_pointer_cast(laplacian_grid.data()), thrust::raw_pointer_cast(x.data()));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return x;

}

//Inner Product Function
//Just doing this so we can have an easier time reading math
Float paracrine::inner_product(thrust::device_vector < Float > A, thrust::device_vector < Float > B) {

  Float result = thrust::inner_product(A.begin(), A.end(), B.begin(), 0.0 f);

  return result;
}

//Conjugate Gradient Function
//TODO: Create CG function in a kernel, not just thrust
//This is currently the slowest part of the code and it's because I didn't create a nice kernel for it yet
thrust::device_vector < Float > paracrine::CG(thrust::device_vector < Float > A, thrust::device_vector < Float > b, thrust::device_vector < Float > grid,
  thrust::device_vector < Float > laplacian_grid, int max_iterations, Float error_tol) {
  //std::cout << "Conjugate Gradient Started" << std::endl;

  //Solves Ax=b (takes in A and b, returns x)
  //A has size 3x3x3 and b has size grid_size*grid_size*grid_size
  //remember that when we multiply A and x, we will get Ax which is grid_size x grid_size x grid_size  (by using our mask_mult function)

  //See B2 in https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf 

  //set counter to 0
  int counter = 0;

  //To determine residual, we need r=b-Ax
  //but we need to find Ax first
  //First guess x: (remember we flatten our matrices)

  //Call initial guess creator ENABLE THIS WHEN DOING LARGER COMPUTATIONS
  //thrust::device_vector<Float> x = initial_guess(grid, laplacian_grid);
  thrust::device_vector < Float > x(grid_size * grid_size * grid_size);
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
    x[i] = (1 - dt * decay) * grid[i] + dt * diffusion * laplacian_grid[i];
  }
  //Laplacian grid(grid_size * grid_size * grid_size) is the grid multiplied by the discrete laplacian with mask_mult function

  //Calculate Ax with our initial guess x
  thrust::device_vector < Float > Ax = mask_mult(x, A); //Ax is now grid_size x grid_size x grid_size

  //Determine residual r = b - Ax
  thrust::device_vector < Float > r(grid_size * grid_size * grid_size);
  thrust::transform(b.begin(), b.end(), Ax.begin(), r.begin(), thrust::minus < Float > ());

  //Take copy of r (MAYBE ERROR HERE)
  thrust::device_vector < Float > d = r;

  Float delta_new = inner_product(r, r);
  Float delta_old = delta_new;
  Float alpha;
  Float beta;
  //thrust::device_vector<Float> alpha_d(grid_size*grid_size*grid_size);//replace with dummy to save space
  //thrust::device_vector<Float> alpha_q(grid_size*grid_size*grid_size);//replace with dummy to save space
  //thrust::device_vector<Float> beta_d(grid_size*grid_size*grid_size);//replace with dummy to save space
  thrust::device_vector < Float > dummy(grid_size * grid_size * grid_size); //replace with dummy to save space
  thrust::device_vector < Float > q(grid_size * grid_size * grid_size);

  //Start loop
  while (counter < max_iterations && delta_new > error_tol) {
    q = mask_mult(d, A);
    alpha = delta_new / (inner_product(d, q));

    //calculate x = x + alpha*d
    //first calculate alpha_d=alpha*d
    thrust::transform(d.begin(), d.end(), dummy.begin(), alpha * _1);
    //now calculate x = x +alpha_d
    thrust::transform(x.begin(), x.end(), dummy.begin(), x.begin(), thrust::plus < Float > ());

    if (counter % 50 == 0) { //recalculate residual to remove floating point error
      //r=b-Ax
      //thrust::device_vector<Float> Ax = mask_mult(x, A);  //calculate Ax
      dummy = mask_mult(x, A); //calculate Ax
      thrust::transform(b.begin(), b.end(), dummy.begin(), r.begin(), thrust::minus < Float > ()); //r=b-Ax
    } else {
      //r=r-alpha*q
      thrust::transform(q.begin(), q.end(), dummy.begin(), alpha * _1); //alpha*q=alpha_q
      thrust::transform(r.begin(), r.end(), dummy.begin(), r.begin(), thrust::minus < Float > ()); //r=r-alpha_q
    }

    delta_old = delta_new;
    delta_new = inner_product(r, r);
    beta = delta_new / delta_old;

    //calculate d=r+beta*d
    thrust::transform(d.begin(), d.end(), dummy.begin(), beta * _1); //beta*d=beta_d
    thrust::transform(r.begin(), r.end(), dummy.begin(), d.begin(), thrust::plus < Float > ()); //d=r+beta_d

    counter += 1;
    std::cout << "Iteration " << counter << " done" << std::endl;
  }
  //std::cout << "Done after " << counter << " iterations" << std::endl;

  return x;

}

//Diffusion Stepper Function
thrust::device_vector < Float > paracrine::diffusion_stepper(thrust::device_vector < Float > grid) {
  //std::cout << "Diffusion Stepper Started" << std::endl;
  //Returns Grid

  //get b so that we can have Ax=b
  b = mask_mult(grid, B); //multiplies grid and B. Grid has size 32x32x32 and mask has size 3x3x3
  //this is element wise multiplication and summing as we slide the mask across B

  //We now have Ax=b and we can solve it with whatever linear solver
  //Choose conjugate gradient

  //Calculate Laplacian Grid
  laplacian_grid = mask_mult(grid, stencil);

  //start conjugate gradient
  int max_iterations = 20;
  float error_tol = 0.000001;
  //A is calculated in initialization()
  grid = CG(A, b, grid, laplacian_grid, max_iterations, error_tol);

  std::cout << "Diffusion Step DONE" << std::endl;

  return grid;
}

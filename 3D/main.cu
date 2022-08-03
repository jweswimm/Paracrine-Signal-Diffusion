#include "paracrine.cuh"

//Tests
void innerproducttest() {
  //Read that thrust inner product sometimes yields weird results
  //so I thought it'd be good to check
  //Get all variables set so we can create paracrine object
  int grid_size = 2;
  int nnz = 1;
  Float dx = 1.0;
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector < Float > neuron_locations_x(nnz);
  thrust::host_vector < Float > neuron_locations_y(nnz);
  thrust::host_vector < Float > neuron_locations_z(nnz);
  thrust::host_vector < Float > neuron_IC(nnz);
  thrust::host_vector < Float > grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
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
  thrust::device_vector < Float > v1(size, 1.0);
  thrust::device_vector < Float > v2(size, 1.0);
  for (int i = 0; i < size; i++) {
    std::cout << "v1[" << i << "]=" << v1[i] << std::endl;
    std::cout << "v2[" << i << "]=" << v2[i] << std::endl;
    std::cout << std::endl;
  }
  std::cout << "Inner product of v1 and v2 is " << IPtest.inner_product(v1, v2);
}

//Conjugate Gradient Test
void CGtest() {
  //REMEMBER TO ENABLE THE FIRST GUESS KERNEL IN THE CG FUNCTION WHEN WE ARE DOING LARGE COMPUTATION
  //Get all variables set so we can create paracrine object
  int grid_size = 4;
  int nnz = 1;
  Float dx = 1.0;
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector < Float > neuron_locations_x(nnz);
  thrust::host_vector < Float > neuron_locations_y(nnz);
  thrust::host_vector < Float > neuron_locations_z(nnz);
  thrust::host_vector < Float > neuron_IC(nnz);
  thrust::host_vector < Float > grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 0.5;
    neuron_locations_y[i] = 0.5;
    neuron_locations_z[i] = 0.5;
  }

  //Set initial conditions on grid
  for (int i = 0; i < grid_size * grid_size * grid_size; i++)
    grid_IC[i] = 1.0;

  //initialize object
  paracrine CGtest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);

  //CG solves Ax=b
  thrust::device_vector < Float > A(27, 1.0);
  thrust::device_vector < Float > b(grid_size * grid_size * grid_size, 10);
  int max_iterations = 1000;
  Float error_tol = 0.000001;

  //Now the answer to Ax=b should be x such that mask_mult(x,A)=b;
  thrust::device_vector < Float > x(grid_size * grid_size * grid_size);

  thrust::device_vector < Float > laplacian_grid = CGtest.mask_mult(grid_IC, A);

  x = CGtest.CG(A, b, grid_IC, laplacian_grid, max_iterations, error_tol);

  //Now we need to test mask_mult(x,A)=b;
  thrust::device_vector < Float > b_test(grid_size * grid_size * grid_size);
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
  thrust::host_vector < Float > neuron_locations_x(nnz);
  thrust::host_vector < Float > neuron_locations_y(nnz);
  thrust::host_vector < Float > neuron_locations_z(nnz);
  thrust::host_vector < Float > neuron_IC(nnz);
  thrust::host_vector < Float > grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
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
  thrust::device_vector < Float > A(27, 1.0);
  thrust::device_vector < Float > x(grid_size * grid_size * grid_size, 1.0);
  thrust::device_vector < Float > b(grid_size * grid_size * grid_size);
  b = MaskMulttest.mask_mult(x, A);

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
  for (int i = 0; i < grid_size; i++) {
    for (int j = 0; j < grid_size; j++) {
      for (int k = 0; k < grid_size; k++) {
        std::cout << "b[" << i << "," << j << "," << k << "]=" << b[grid_size * grid_size * i + grid_size * j + k] << std::endl;
      }
    }
  }
}

//Interpolation Test
void Interpolationtest() {
  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  Float dx = 1.0;
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector < Float > neuron_locations_x(nnz);
  thrust::host_vector < Float > neuron_locations_y(nnz);
  thrust::host_vector < Float > neuron_locations_z(nnz);
  thrust::host_vector < Float > neuron_IC(nnz);
  thrust::host_vector < Float > grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 0.5;
    neuron_locations_y[i] = 0.5;
    neuron_locations_z[i] = 0.5;
  }

  //set neuron locations
  //First Neuron
  neuron_locations_x[0] = 0.5;
  neuron_locations_y[0] = 0.5;
  neuron_locations_z[0] = 0.5;

  //Second Neuron
  neuron_locations_x[1] = 0.25;
  neuron_locations_y[1] = 0.25;
  neuron_locations_z[1] = 0.25;

  //Third Neuron
  neuron_locations_x[2] = 0.75;
  neuron_locations_y[2] = 0.75;
  neuron_locations_z[2] = 0.75;

  //Set initial conditions on grid
  for (int i = 0; i < grid_size * grid_size * grid_size; i++)
    grid_IC[i] = 10;

  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 0] = 0; //grid[0,0,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 0] = 0; //grid[1,0,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 0] = 0; //grid[0,1,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 0] = 0; //grid[1,1,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 1] = 0; //grid[0,0,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 1] = 0; //grid[1,0,1]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 1] = 0; //grid[0,1,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 1] = 1; //grid[1,1,1]

  //initialize object
  paracrine Interpolationtest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  Interpolationtest.initialize();

  //Create Concentration at Neuron Locations vector
  thrust::device_vector < Float > neuron_concentrations(nnz);
  neuron_concentrations = Interpolationtest.interpolate(nnz, grid_size, grid_IC);

  for (int i = 0; i < 3; i++) {
    std::cout << "Neuron " << i << " concentration at location [" << neuron_locations_x[i] <<
      "," << neuron_locations_y[i] << "," << neuron_locations_z[i] << "]=" << neuron_concentrations[i] << std::endl;
  }

}

//Spreading Test
void Spreadtest() {
  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  Float dx = 1.0;
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector < Float > neuron_locations_x(nnz);
  thrust::host_vector < Float > neuron_locations_y(nnz);
  thrust::host_vector < Float > neuron_locations_z(nnz);
  thrust::host_vector < Float > neuron_IC(nnz);
  thrust::host_vector < Float > grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) { //get every neuron away from beginning cube
    neuron_locations_x[i] = 20;
    neuron_locations_y[i] = 20;
    neuron_locations_z[i] = 20;
  }

  //set 3 test neuron locations to beginning cube
  //First Neuron
  neuron_locations_x[0] = 0.5;
  neuron_locations_y[0] = 0.5;
  neuron_locations_z[0] = 0.5;

  //Second Neuron
  neuron_locations_x[1] = 0.5;
  neuron_locations_y[1] = 0.5;
  neuron_locations_z[1] = 0.5;

  //Third Neuron
  neuron_locations_x[2] = 0.5;
  neuron_locations_y[2] = 0.5;
  neuron_locations_z[2] = 0.5;

  //declare dummy neuron_concentrations vector (NOT USED RIGHT NOW)
  // this neuron_concentrations vector is added so that when we want to have 
  // the concentration at the neuron locations dictate how much neurotransmitter is released, we can easily do it
  // for now, we are just releasing generation_constant amount of neurotransmitter (a constant)
  thrust::device_vector < Float > neuron_concentrations(nnz);

  //Set initial conditions on grid and make a dummy vector for neuron_concentrations
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
    grid_IC[i] = 10;
  }
  for (int i = 0; i < nnz; i++)
    neuron_concentrations[i] = 10;

  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 0] = 0; //grid[0,0,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 0] = 0; //grid[1,0,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 0] = 0; //grid[0,1,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 0] = 0; //grid[1,1,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 1] = 0; //grid[0,0,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 1] = 0; //grid[1,0,1]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 1] = 0; //grid[0,1,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 1] = 0; //grid[1,1,1]

  //initialize object
  paracrine Spreadingtest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  Spreadingtest.initialize();

  Float generation_constant = 1;

  std::cout << "Before Spreading:" << std::endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        std::cout << "grid[" << i << "," << j << "," << k << "]=" 
		<< grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
      }
    }
  }
  grid_IC = Spreadingtest.spread(generation_constant, grid_IC, neuron_concentrations);
  std::cout << "After Spreading " << generation_constant << " from each of the three neurons" << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << "Neuron[" << i << "] at location [" << neuron_locations_x[i] << "," << neuron_locations_y[i] <<
      "," << neuron_locations_z[i] << "]" << std::endl;
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        std::cout << "grid[" << i << "," << j << "," << k << "]="
		<< grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
      }
    }
  }

}

//Test Diffusion
void Diffusiontest() {
  std::cout << "Testing Diffusion" << std::endl;
  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  Float dx = 1.0;
  Float dt = 0.04;
  Float diffusion = 2e-5;
  Float decay = 3e-2;
  thrust::host_vector < Float > neuron_locations_x(nnz);
  thrust::host_vector < Float > neuron_locations_y(nnz);
  thrust::host_vector < Float > neuron_locations_z(nnz);
  thrust::host_vector < Float > neuron_IC(nnz);
  thrust::host_vector < Float > grid(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 0.5;
    neuron_locations_y[i] = 0.5;
    neuron_locations_z[i] = 0.5;
  }

  //Set initial conditions on grid
  for (int i = 0; i < grid_size * grid_size * grid_size; i++)
    grid[i] = 1.0;

  //initialize object
  paracrine Diffusiontest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid, neuron_IC);
  Diffusiontest.initialize();

  std::cout << "Initialization finished" << std::endl;
  //Open Document
  std::ofstream output;
  output.open("diffusiontest2.txt");

  //How many seconds would you like the simulation to run for?
  Float time = 10;
  //Get number of diffusion steps
  int total_steps = ceil(time / dt);

  for (int timestep = 0; timestep < total_steps; timestep++) {
    grid = Diffusiontest.diffusion_stepper(grid);

    if (timestep % (total_steps / 100) == 0)
      std::cout << 100 * (Float(timestep) / Float(total_steps)) << "%" << std::endl;

    output << timestep * dt << " " << grid[grid_size * grid_size * 10 + grid_size * 10 + 10] << std::endl;
  }
  output.close();

}

int main() {
  //	innerproducttest();
  CGtest();
  //	MaskMulttest();
  //	Interpolationtest();
  //	Spreadtest();
  //	Diffusiontest();

}

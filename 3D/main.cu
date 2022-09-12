#include "paracrine.cuh"
#include <fstream>
#include <chrono>
#include <cstdlib>

using namespace std::chrono;
using namespace thrust::placeholders;

//All together now!
//paracrine_test brings together the spreading, diffusion stepper, and interpolation
thrust::device_vector<Float> paracrine_test() {
    //Start with concentration at neuron location
    //spread to grid 
    //step the diffusion equation
    //interpolate back to neuron
    //return neuron_concentrations

    //Seed random number
    srand((unsigned)time(0));
    
    //Get all variables set so we can create paracrine object
    int grid_size = 32; // total size of grid = (grid_size x grid_size x grid_size)
    int nnz = 10000; //number of neurons
    Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
    Float dt = 0.04; //seconds     (so 40 milliseconds)
    Float diffusion = 2e-5;  //diffusion constant taken from Ningyuan's code
    Float decay = 3e-4; //difusion constant taken from Ningyuan's code

    //Create empty thrust vectors
    thrust::host_vector <Float> neuron_locations_x(nnz); //x coordinate locations of neurons
    thrust::host_vector <Float> neuron_locations_y(nnz); //y coordinate locations of neurons
    thrust::host_vector <Float> neuron_locations_z(nnz); //z coordinate locations of neurons
    thrust::device_vector <Float> neuron_concentrations(nnz); //concentrations at neuron locations
    thrust::device_vector <Float> grid(grid_size * grid_size * grid_size); //concentrations at grid point locations
    for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
        grid[i] = ((Float)rand() / (Float)RAND_MAX) * 2;
    }


    //Neuron Initialization
    //Note: The location of gridpoints and neurons is on a scale from 0 to 31
    //Neuron at location (0.5, 0.5, 0.5) is at PHYSICAL location (0.5 * dx, 0.5 * dx, 0.5 * dx) 
    //where dx is the distance between gridpoints
    //For this code, it simplifies calculations and programming to have locations in the code be from 0 to 31)
    for (int i = 0; i < nnz; i++) {
        //Create random neuron locations
        neuron_locations_x[i] = ((Float)rand() / (Float)RAND_MAX) * 31;
        neuron_locations_y[i] = ((Float)rand() / (Float)RAND_MAX) * 31;
        neuron_locations_z[i] = ((Float)rand() / (Float)RAND_MAX) * 31;

        //Create some random concentrations at neuron locations
        neuron_concentrations[i] = ((Float)rand() / (Float)RAND_MAX); //values between 0 and 1
    }

    //Create paracrine object
    paracrine p_obj(grid_size, nnz, dx, dt,
     diffusion, decay, neuron_locations_x,
     neuron_locations_y, neuron_locations_z, grid, neuron_concentrations);

    //Initialize object
    p_obj.initialize();
    //Open Document
    std::ofstream output;
    output.open("paracrinetest.txt");

    //How many seconds would you like the simulation to run for?
    Float time = 100;
    //Get number of diffusion steps
    int total_steps = ceil(time / dt);

    for (int timestep = 0; timestep < total_steps; timestep++) {
        if (timestep % (total_steps / 100) == 0)
            std::cout << 100 * (Float(timestep) / Float(total_steps)) << "%" << std::endl;

        //Get concentration on grid from neuron concentrations
        grid = p_obj.spread(0.001, grid, neuron_concentrations);

        //Diffusion step
        grid = p_obj.update_density(grid);
        output << timestep * dt << "," << grid[grid_size * grid_size * 16 + grid_size * 16 + 16] << std::endl;

        //Interpolation step
        neuron_concentrations = p_obj.interpolate(grid);

    }
    output.close();
    std::cout << "DONE" << std::endl;
}


//Create test to spread from neuron to grid, then interpolate from grid to neuron
//we should return back to the same values
void spread_interp_test() {
    std::cout << "Testing spreading, then interpolation" << std::endl;

  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  //Float dx = 1.0;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::device_vector <Float> neuron_IC(nnz);
  thrust::device_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  //Locations: between 0 and grid_size (e.g. 0 and <32), x=0.5 would be halfway between the first and second gridpoint
  //the actual location of (0.5,0.5,0.5) is (0.5*dx, 0.5*dx, 0.5*dx) but for ease in computation, we leave as is and 
  //account for the real distances in the initialization step (paracrine.cu initialization function)
  for (int i = 0; i < nnz; i++) { //get every neuron away from beginning cube
    neuron_locations_x[i] = 20;
    neuron_locations_y[i] = 20;
    neuron_locations_z[i] = 20;
  }

  //set 3 test neuron locations to beginning cube
  //First Neuron
  neuron_locations_x[0] = 0.1;
  neuron_locations_y[0] = 0.1;
  neuron_locations_z[0] = 0.1;

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
  thrust::device_vector <Float> neuron_concentrations(nnz);

  //Set initial conditions on grid and make a dummy vector for neuron_concentrations
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
    grid_IC[i] = 0;
  }
  for (int i = 0; i < nnz; i++) {
      //neuron_concentrations[i] = 0.0;
//      neuron_concentrations[i] = ((Float)rand() / (Float)RAND_MAX) * 2;
      neuron_concentrations[i] =1.0;

  }
//  neuron_concentrations[1] = 100;

  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 0] = 0; //grid[0,0,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 0] = 0; //grid[1,0,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 0] = 0; //grid[0,1,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 0] = 0; //grid[1,1,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 1] = 0; //grid[0,0,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 1] = 0; //grid[1,0,1]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 1] = 0; //grid[0,1,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 1] = 0; //grid[1,1,1]

  //initialize object
  paracrine test(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  test.initialize();

  //Generation constant is how much neurotransmitter is being spread from each neuron
  //It's incorporated as a large vector to have different neurotransmitter creation values
  //per neuron, see paracrine.cu spread function (particularly where P is created)
  Float generation_constant = 0;

  thrust::device_vector<Float> testvec(grid_size * grid_size * grid_size,0);

  std::cout << "Before Spreading:" << std::endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        std::cout << "grid[" << i << "," << j << "," << k << "]=" 
		<< grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
      }
    }
  }
  for (int n = 0; n < 4; n++) {
      //The spreading function takes the amount generated from the neuron location and adds it to the already
      //existing concentration at the gridpoint location
      //so this test starts with some initial neuron concentration
      //spreads it to the grid
      testvec = test.spread(generation_constant, grid_IC, neuron_concentrations);

      thrust::transform(testvec.begin(), testvec.end(), grid_IC.begin(), grid_IC.begin(), thrust::minus <Float> ());
//      for (int f = 0; f < grid_size * grid_size * grid_size; f++)
 //         grid_IC[f] = testvec[f];
      //std::cout << "After Spreading " << generation_constant << " from each of the three neurons" << std::endl;
      std::cout << "After Spreading the neuron concentration from each of the three neurons" << std::endl;
      for (int i = 0; i < 3; i++) {
          std::cout << "Neuron[" << i << "] at location [" << neuron_locations_x[i] << "," << neuron_locations_y[i] <<
              "," << neuron_locations_z[i] << "] ="<<neuron_concentrations[i]<< std::endl;
      }

      Float sum = 0;
      for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
              for (int k = 0; k < 2; k++) {
                  std::cout << "grid[" << i << "," << j << "," << k << "]="
                      << grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
                  sum = sum + grid_IC[grid_size * grid_size * i + grid_size * j + k];
              }
          }
      }
      std::cout << "SUM:" << sum << std::endl;

      neuron_concentrations = test.interpolate(grid_IC);

      sum = 0;
      for (int i = 0; i < 3; i++) {
          std::cout << "Neuron " << i << " concentration at location [" << neuron_locations_x[i] <<
              "," << neuron_locations_y[i] << "," << neuron_locations_z[i] << "]=" << neuron_concentrations[i] << std::endl;
          sum = sum + neuron_concentrations[i];
      }
  }
}

void update_density_test() {
  std::cout << "Testing Update Density" << std::endl;
  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 0.04;
  Float diffusion = 2e-5;
  //Float diffusion = 2e-3;
//  Float diffusion = 2e-1;
  Float decay = 3e-4;
//  Float decay = 3e-4;
//  Float decay = 10;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::host_vector <Float> neuron_IC(nnz);
  thrust::host_vector <Float> grid(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 0.5;
    neuron_locations_y[i] = 0.5;
    neuron_locations_z[i] = 0.5;
  }

  //Set initial conditions on grid
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
      //grid[i] = 1e-4;
  //    grid[i] = (rand() % 100) / 100;
      grid[i] = 0.001;
  }
      grid[grid_size * grid_size * 16 + grid_size * 16 + 16] = 1.0;

  //initialize object
  paracrine Diffusiontest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid, neuron_IC);
  Diffusiontest.initialize();

  std::cout << "Initialization finished" << std::endl;
  //Open Document
  std::ofstream output;
  output.open("pdiff.txt");

  //How many seconds would you like the simulation to run for?
  Float time = 100;
  //Get number of diffusion steps
  int total_steps = ceil(time / dt);

  for (int timestep = 0; timestep < total_steps; timestep++) {
    grid = Diffusiontest.update_density(grid);

    if (timestep % (total_steps / 100) == 0)
      std::cout << 100 * (Float(timestep) / Float(total_steps)) << "%" << std::endl;


//    output << timestep * dt << ",";
    //Plot x plane
//    for (int i = 0; i < grid_size; i++) {
 //       if (i == grid_size - 1)
  //          output << grid[grid_size * grid_size * i + grid_size * 16 + 16] << std::endl;
   //     else 
    //    output << grid[grid_size * grid_size * i + grid_size * 16 + 16] << ",";
    //}
//            output << timestep * dt << "," <<grid[grid_size * grid_size * 1 + grid_size * 1 + 2] << std::endl;

    if (timestep == total_steps -5) {
        for (int i = 0; i < grid_size; i++) {
            output << i << "," << grid[grid_size * grid_size * 16 + grid_size * 16 + i] << std::endl;
        }
    }
  }
//  output.close();
  std::cout << "Entire Diffusion Done" << std::endl;
  std::cout << total_steps - 5;
  output.close();
}


//Conjugate Gradient Test
void CGtest() {
  //REMEMBER TO ENABLE THE FIRST GUESS KERNEL IN THE CG FUNCTION WHEN WE ARE DOING LARGE COMPUTATION
  //Get all variables set so we can create paracrine object
  int grid_size = 4;
  int nnz = 1;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::host_vector <Float> neuron_IC(nnz);
  thrust::host_vector <Float> grid_IC(grid_size * grid_size * grid_size);

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
  //thrust::device_vector <Float> A(27, 1.0);
  thrust::device_vector <Float> A(27,0.0);
  //If main diagonal
  A[3 * 3 * 0 + 3 * 0 + 0] = 3.0; //A[0,0,0]=1
  A[3 * 3 * 1 + 3 * 1 + 1] = 3.0; //A[1,1,1]=2
  A[3 * 3 * 2 + 3 * 2 + 2] = 3.0; //A[2,2,2]=3
  //If a vertex, set equal to 1
  A[3 * 3 * 0 + 3 * 2 + 2] = 1.0; //A[0,2,2]=1
  A[3 * 3 * 2 + 3 * 0 + 2] = 1.0; //A[2,0,2]=1
  A[3 * 3 * 2 + 3 * 2 + 0] = 1.0; //A[2,2,0]=1
  A[3 * 3 * 2 + 3 * 0 + 0] = 1.0; //A[2,0,0]=1
  A[3 * 3 * 0 + 3 * 0 + 2] = 1.0; //A[0,0,2]=1
  A[3 * 3 * 0 + 3 * 2 + 0] = 1.0; //A[0,2,0]=1
  
  //thrust::device_vector <Float> b(grid_size * grid_size * grid_size, 10);
  thrust::device_vector <Float> b(grid_size * grid_size * grid_size);
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
      b[i] = i;
  }

  //Now the answer to Ax=b should be x such that convolve(x,A)=b;
  thrust::device_vector <Float> x(grid_size * grid_size * grid_size);

  thrust::device_vector <Float> laplacian_grid = CGtest.convolve(grid_IC, A);

  x = CGtest.CG(A, b);

  //Now we need to test convolve(x,A)=b;
  thrust::device_vector <Float> b_test(grid_size * grid_size * grid_size);
  b_test = CGtest.convolve(x, A);
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
    std::cout << "real b=" << b[i] << "   b_test=" << b_test[i] << std::endl;
  }

}

//Mask Multiplication Test
void laplaciantest() {

  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::host_vector <Float> neuron_IC(nnz);
  thrust::device_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 0.5;
    neuron_locations_y[i] = 0.5;
    neuron_locations_z[i] = 0.5;
  }

  //Set initial conditions on grid
//  for (int i = 0; i < grid_size * grid_size * grid_size; i++)
 //   grid_IC[i] = rand();

  thrust::device_vector<Float> real_values(grid_size*grid_size*grid_size);

  for (int i = 0; i < grid_size; i++) {
      Float xx = dx * i;
      for (int j = 0; j < grid_size; j++) {
          Float yy = dx * j;
          for (int k = 0; k < grid_size; k++) {
            Float zz = dx * k;
            grid_IC[grid_size * grid_size * i + grid_size * j + k] = xx * xx * xx + yy * yy + zz * zz + xx * yy * zz;
            real_values[grid_size * grid_size * i + grid_size * j + k] = 6*xx+4;

          }
      }
  }



  //initialize object
  paracrine MaskMulttest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  MaskMulttest.initialize();

  thrust::device_vector<Float> result(grid_size*grid_size*grid_size);
  thrust::device_vector<Float> laplace_stencil_divided(27);
  laplace_stencil_divided = MaskMulttest.stencil;
  Float dum = 1 / (dx * dx * 30.0);
  thrust::transform(laplace_stencil_divided.begin(), laplace_stencil_divided.end(), laplace_stencil_divided.begin(), dum * _1);
  result = MaskMulttest.convolve(grid_IC, laplace_stencil_divided);

  //For this test, given A (3x3x3) and x (grid_size x grid_size x grid_size), we should be able to compute Ax=b
  //This is done by sliding A across x and calculating the sum of the element wise multiplications
//  thrust::device_vector <Float> A(27, 1.0);
//  thrust::device_vector <Float> x(grid_size * grid_size * grid_size, 1.0);
//  thrust::device_vector <Float> b(grid_size * grid_size * grid_size);
//  b = MaskMulttest.convolve(x, A);

  //The following comments were meant for the test case of a grid of 1's
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
  //-------------------------------------------------------------------

  std::cout << "Testing laplacian of f(x,y,z)=x^2+y^2+z^2+xyz, the result should be laplacian_f = 6 at values not touching the boundary"
      << std::endl;
  
  for (int i = 1; i < 5; i++) {
    for (int j = 1; j < 5; j++) {
      for (int k = 1; k < 5; k++) {
        std::cout << "expected_values[" << i << "," << j << "," << k << "]=" << real_values[grid_size * grid_size * i + grid_size * j + k] << "     ";
        std::cout << "values[" << i << "," << j << "," << k << "]=" << result[grid_size * grid_size * i + grid_size * j + k] << std::endl;
      }
    }
  }
}

//Interpolation Test
void Interpolationtest() {
  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::device_vector <Float> neuron_IC(nnz);
  thrust::device_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 30;
    neuron_locations_y[i] = 30;
    neuron_locations_z[i] = 30;
  }

  //set neuron locations
  //First Neuron
  neuron_locations_x[0] = 0.1;
  neuron_locations_y[0] = 0.1;
  neuron_locations_z[0] = 0.1;

  //Second Neuron
  neuron_locations_x[1] = 0.5;
  neuron_locations_y[1] = 0.5;
  neuron_locations_z[1] = 0.5;

  //Third Neuron
  neuron_locations_x[2] = 0.5;
  neuron_locations_y[2] = 0.5;
  neuron_locations_z[2] = 0.5;

  //Set initial conditions on grid
  for (int i = 0; i < grid_size * grid_size * grid_size; i++)
    grid_IC[i] = 10;

  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 0] = 0.0; //grid[0,0,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 0] = 0.0 ; //grid[1,0,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 0] = 0.0; //grid[0,1,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 0] = 0.0; //grid[1,1,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 1] = 0.0; //grid[0,0,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 1] = 0.2; //grid[1,0,1]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 1] = 0.0; //grid[0,1,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 1] = 1.0; //grid[1,1,1]

  //initialize object
  paracrine Interpolationtest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  Interpolationtest.initialize();

  //Create Concentration at Neuron Locations vector
  thrust::device_vector <Float> neuron_concentrations(nnz);
  neuron_concentrations = Interpolationtest.interpolate( grid_IC);

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
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::host_vector <Float> neuron_IC(nnz);
  thrust::host_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  //Locations: between 0 and grid_size (e.g. 0 and <32), x=0.5 would be halfway between the first and second gridpoint
  //the actual location of (0.5,0.5,0.5) is (0.5*dx, 0.5*dx, 0.5*dx) but for ease in computation, we leave as is and 
  //account for the real distances in the initialization step (paracrine.cu initialization function)
  for (int i = 0; i < nnz; i++) { //get every neuron away from beginning cube
    neuron_locations_x[i] = 20;
    neuron_locations_y[i] = 20;
    neuron_locations_z[i] = 20;
  }

  //set 3 test neuron locations to beginning cube
  //First Neuron
  neuron_locations_x[0] = 0.25;
  neuron_locations_y[0] = 0.25;
  neuron_locations_z[0] = 0.25;

  //Second Neuron
  neuron_locations_x[1] = 0.25;
  neuron_locations_y[1] = 0.25;
  neuron_locations_z[1] = 0.25;

  //Third Neuron
  neuron_locations_x[2] = 0.25;
  neuron_locations_y[2] = 0.25;
  neuron_locations_z[2] = 0.25;

  //declare dummy neuron_concentrations vector (NOT USED RIGHT NOW)
  // this neuron_concentrations vector is added so that when we want to have 
  // the concentration at the neuron locations dictate how much neurotransmitter is released, we can easily do it
  // for now, we are just releasing generation_constant amount of neurotransmitter (a constant)
  thrust::device_vector <Float> neuron_concentrations(nnz);

  //Set initial conditions on grid and make a dummy vector for neuron_concentrations
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
    grid_IC[i] = 0;
  }
  for (int i = 0; i < nnz; i++)
    neuron_concentrations[i] = 10;

//  neuron_concentrations[1] = 100;

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

  //Generation constant is how much neurotransmitter is being spread from each neuron
  //It's incorporated as a large vector to have different neurotransmitter creation values
  //per neuron, see paracrine.cu spread function (particularly where P is created)
  Float generation_constant = 2;

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

  Float sum = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        std::cout << "grid[" << i << "," << j << "," << k << "]="
		<< grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
        sum = sum + grid_IC[grid_size * grid_size * i + grid_size * j + k];
      }
    }
  }
  std::cout << "SUM:" << sum << std::endl;
}

int main() {
    //Timing
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();

    //Tests------------------
    //CGtest();
    //laplaciantest();
  	//Interpolationtest();
  	Spreadtest();   
    //spread_interp_test();
    //update_density_test();
    //paracrine_test();
    //-----------------------


    //Timing
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout <<"Entire simulation took "<< ms_int.count() << "ms\n";

}

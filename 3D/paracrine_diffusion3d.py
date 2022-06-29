#Author: Joe Wimmergren 2022
import torch
import numpy as np
import torch.nn.functional as F

torch.__version__
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #needed for matplotlib for this computer (duplicate files)
#https://github.com/matplotlib/matplotlib/issues/21513

#Turn off autograd
torch.set_grad_enabled(False)

#Reproduce function
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
#seed_everything(42)



def CG(A,b,grid,grid_size,max_iterations,error_tol,BC,diffusion,decay,laplacian_grid,dt):
    #Conjugate Gradient Function
    #Define padding function
    #use this after each convolution to pad the grid back with the boundary condition
    #pad=torch.nn.ConstantPad3d(1,BC)
    
    
    #see https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf 
    #B2 from the pdf above
    #A has size 3,3,3
    #b has size grid_size,grid_size,grid_size
    #Set counter to 0
    i=0
    
    #To determine residual, we need r=b-Ax
    #but we need to find Ax first
    #First guess x:
    #x=torch.rand(1,1,grid_size,grid_size,grid_size,device='cuda',dtype=float) #random first guess
    #x=b.clone() #grid as first guess
    
     
    x = (1 - dt * decay) \
    * grid + dt * diffusion * laplacian_grid #laplacian_grid is the stencil (3x3x3) convolved
    #with the grid (grid_size,grid_size,grid_size) 
    #to yield laplacian_grid (grid_size,grid_size,grid_size)
    #This is a good first guess, should be close to the solution (gambling that computing the above
    #value of x is faster than the multiple iterations required to get to a solution from pure CG solver
    #from a random set of numbers)
    
    
    #A and x cannot be multiplied normally, we must use a convolution to slide A across x
    #resize A for the incoming convolution
    A=A.view(1,1,3,3,3)
    Ax=F.conv3d(x,A,padding=1) #Ax now has size 1,1,grid_size,grid_size,grid_size
    #resize b for the incoming subtraction
    b=b.view(1,1,grid_size,grid_size,grid_size)
    
    #Determine residual
    r=b.clone()-Ax.clone() #size=1,1,grid_size,grid_size
    #print("R size:", r.size())
    
    #Take copy of residual
    d=r.clone() #size=1,1,grid_size,grid_size,grid_size
    
    #Set delta_new=rTr
    delta_new=torch.inner(r.view(grid_size**3),r.view(grid_size**3)) #delta_new is now a scalar
    
    
    #set delta
    delta_old=delta_new.clone()
    
    #start loop
    while (i<max_iterations) and (delta_new.item()>error_tol):
        q=F.conv3d(d,A,padding=1) #size=1,1,grid_size,grid_size,grid_size
        alpha=delta_new/(torch.inner(d.view(grid_size**3),q.view(grid_size**3))) #scalar
        x=x+alpha.clone()*d.clone() #size=1,1,grid_size,grid_size,grid_size
        
        if (i%50==0): #recalculate residual to remove floating point error
            r=b.clone()-(F.conv3d(x,A,padding=1))
            
        else:
            r=r-alpha.clone()*q.clone()
        
        delta_old=delta_new.clone()
        delta_new=torch.inner(r.view(grid_size**3),r.view(grid_size**3))
        beta=delta_new.clone()/delta_old.clone()
        d=r.clone()+beta.clone()*d.clone()
        i=i+1
        #print(delta_new)
    if (i>18):
        print("Done after ",i,"iterations") #just to see if it's taking a lot of iterations
    return torch.abs(x.view(grid_size,grid_size,grid_size))


class paracrine: #original use of this code is for neurotransmitters called paracrine factors
    #Should maybe call this class something more intuitive for other users outside of neuroscience
    def __init__(self,nnz, grid_size, neuron_x, neuron_y, neuron_z, generation_fraction, BC):
        #nnz: number of neurons
        #grid_size: size of grid in each direction, e.g. grid_size=32 means 32x32x32
        #grid: concentration of neurotransmitter at each gridpoint
        #neuron_x: x coordinates of neuron locations
        #neuron_y: y coordinates of neuron locations
        #neuron_z: z coordinates of neuron locations
        #Establish grid size
        global grid_size_x, grid_size_y, grid_size_z #FIX THIS
        grid_size_x=grid_size
        grid_size_y=grid_size
        grid_size_z=grid_size
        #print("Grid size is :", grid_size_x,"x",grid_size_y,"x",grid_size_z)
        
        #Define pad for keeping boundary condition
        global pad
        pad=torch.nn.ConstantPad3d(1,BC)
        
        
        #See https://spie.org/samples/PM159.pdf
        #Calculate Q matrix
        #Determine closest integer below and above (to find gridpoint locations) for each neuron
        global x0, x1, y0, y1, z0, z1, one
        x0=torch.floor(neuron_x).long()
        x1=torch.ceil(neuron_x).long()
        y0=torch.floor(neuron_y).long()
        y1=torch.ceil(neuron_y).long()
        z0=torch.floor(neuron_z).long()
        z1=torch.ceil(neuron_z).long()
        one=torch.ones(neuron_x.size(),device='cuda')


        #Determine distances in all directions for each neuron
        del_x=(neuron_x-x0)/(x1-x0)
        del_y=(neuron_y-y0)/(y1-y0)
        del_z=(neuron_z-z0)/(z1-z0)
        
        global weighted_spread
        weighted_spread=torch.empty(8,nnz,device='cuda',dtype=float)
        if (nnz==1):
            weighted_spread=torch.tensor([(1-del_x)*(1-del_y)*(1-del_z),(1-del_x)*(1-del_y)*(del_z),\
                                             (1-del_x)*(del_y)*(1-del_z),(1-del_x)*(del_y)*(del_z),\
                                             (del_x)*(1-del_y)*(1-del_z),(del_x)*(1-del_y)*(del_z),\
                                             (del_x)*(del_y)*(1-del_z),(del_x)*(del_y)*(del_z)],device='cuda',dtype=float)

        else:
            #for i in range (0,nnz): #PARALLELIZE THIS 
            #weighted_spread[:,i]=torch.stack([(one[i]-del_x[i])*(one[i]-del_y[i])*(one[i]-del_z[i]),(one[i]-del_x[i])*(one[i]-del_y[i])*(del_z[i]),\
                                            #(one[i]-del_x[i])*(del_y[i])*(one[i]-del_z[i]),(one[i]-del_x[i])*(del_y[i])*(del_z[i]),\
                                            #(del_x[i])*(one[i]-del_y[i])*(one[i]-del_z[i]),(del_x[i])*(one[i]-del_y[i])*(del_z[i]),\
                                            #(del_x[i])*(del_y[i])*(one[i]-del_z[i]),(del_x[i])*(del_y[i])*(del_z[i])],1)
            weighted_spread=torch.stack([(one-del_x)*(one-del_y)*(one-del_z),(one-del_x)*(one-del_y)*(del_z),\
                                            (one-del_x)*(del_y)*(one-del_z),(one-del_x)*(del_y)*(del_z),\
                                            (del_x)*(one-del_y)*(one-del_z),(del_x)*(one-del_y)*(del_z),\
                                            (del_x)*(del_y)*(one-del_z),(del_x)*(del_y)*(del_z)],0)
            print(weighted_spread.size())

        #Build distance matrix Q
        global Q
        Q=torch.stack([one,del_x,del_y,del_z,del_x*del_y,del_y*del_z,del_z*del_x, del_x*del_y*del_z],0)
        #print("Q matrix (distance matrix) has been built and it has size", Q.size())

        #Build B matrix and B_inverse
        #global B_inv
        B=torch.tensor([[1,0,0,0,0,0,0,0],[-1,0,0,0,1,0,0,0],[-1,0,1,0,0,0,0,0],\
                           [-1,1,0,0,0,0,0,0],[1,0,-1,0,-1,0,1,0],[1,-1,-1,1,0,0,0,0],\
                           [1,-1,0,0,-1,1,0,0],[-1,1,1,-1,1,-1,-1,1]],device='cuda',dtype=float)
        #B_inv=torch.inverse(B.clone())
        #global LHS
        #LHS=(1/(torch.linalg.norm(Q,ord=2)**2))*torch.matmul(B_inv,Q) 
        print("Paracrine initialization done")
        
   #Interpolation and Spreading-------------------------------------------------------------

    def grid_to_neuron(self,grid): #This matrix has to be done every timestep
        #Build coefficient matrix
        #There will be c values for each element, e.g nnz many c0 values
        #Note, have to use long instead of int for x0, y0, z0 (in order to use tensors as indices)
        #Note: this must be computed every time the interpolation takes place, as the concentration
        #at the grid points will be changing (i.e. grid itself will be changing)
        c0=grid[x0,y0,z0]
        c1=grid[x1,y0,z0]-grid[x0,y0,z0]
        c2=grid[x0,y1,z0]-grid[x0,y0,z0]
        c3=grid[x0,y0,z1]-grid[x0,y0,z0]
        c4=grid[x1,y1,z0]-grid[x0,y1,z0]-grid[x1,y0,z0]+grid[x0,y0,z0]
        c5=grid[x0,y1,z1]-grid[x0,y0,z1]-grid[x0,y1,z0]+grid[x0,y0,z0]
        c6=grid[x1,y0,z1]-grid[x0,y0,z1]-grid[x1,y0,z0]+grid[x0,y0,z0]
        c7=grid[x1,y1,z1]-grid[x0,y1,z1]-grid[x1,y0,z1]-grid[x1,y1,z0]+grid[x1,y0,z0]+grid[x0,y0,z1]+grid[x0,y1,z0]-grid[x0,y0,z0]
        C=torch.stack([c0, c1, c2, c3, c4, c5, c6, c7],0)
        #print ("C0 has size", c0.size())
        #print ("C matrix (coefficient matrix) has been built and it has size", C.size())

        #Finally, calculate output vector that has concentrations at neuron locations
        neuron_concentrations=torch.zeros(nnz,device='cuda')

        if (nnz==1):
            neuron_concentrations=torch.inner(C,Q)
        else:
            #C=torch.transpose(C)
            #for i in range (0,nnz):#PARALLELIZE THIS (shouldn't need loop)
                #neuron_concentrations[i]=torch.inner(C[:,i],Q[:,i],)
            neuron_concentrations=torch.diag(torch.matmul(torch.transpose(C,0,1),Q))

        #print("Inner product is done and concentrations at neurons have been calculated")
        return neuron_concentrations

    
    def production_and_neuron_to_grid(self,neuron_concentrations,grid): #This is executed at every timestep
        #We have to be careful, because there may be two neurons that share one or more gridpoints
        #in that case, we must sum the concentrations at the shared gridpoints after interpolation
        dum_grid=torch.zeros(grid_size_x, grid_size_y, grid_size_z,device='cuda',dtype=float)
        #print(generation_fraction)
        
        #We can use 
        #(a/|Q|^2) * B_inv * Q=P
        #Where a is the neuron concentration
        #Q is the distance matrix
        #B_inv is the inverse of the B matrix of 1's and -1's
        #and P=[grid[x0,y0,z0], grid[x0,y0,z1], grid[x0,y1,z0], grid[x0,y1,z1], grid[x1,y0,z0],\
        #grid[x1,y0,z1], grid[x1,y1,z0], grid[x1,y1,z1]]
        #We are solving for P in this function!
        #We have to do this every timestep because the concentration at the neuron location
        #will change at every timestep, but Q and B won't (which is why we calculated them with
        #the constructor above)
        #In fact, we can calculate (1/|Q|^2)* B_inv * Q ahead of time
        
        
        #Now at every timestep, we have a new vector of neuron_concentrations
        #P=torch.zeros(8,nnz) #8 gridpoints around each neuron, nnz many neurons
        #Neuron concentrations are found from the prod_rate in the class above
        #We take generation_fraction*neuron concentration and this will be the production
        #of neurotransmitter every timestep
        
        #new_neuron_concentrations=neuron_concentrations.clone()+generation_fraction 
        #generation=generation_fraction*neuron_concentrations.clone() #ratio
        generation=generation_fraction#*neuron_concentrations.clone() #constant
        #was originally neuron_concentrations*generation_fraction
                                    
        
        #P=generation.clone()*LHS.clone() #element wise multiplication
        P=generation*weighted_spread.clone()
        #print("Adding ", neuron_concentrations*generation_fraction,"to each neuron")
        #We now have P, the 8 x nnz matrix of concentrations spread from the neurons
        #We now have to fix the issue of if any of the neurons have the same gridpoints

 
        
        #I don't think we need the dum_grid, will speed up 
        
        if (nnz==1): #for test case
            grid[x0,y0,z0]=P[0].clone()+grid[x0,y0,z0]
            grid[x0,y0,z1]=P[1].clone()+grid[x0,y0,z1]
            grid[x0,y1,z0]=P[2].clone()+grid[x0,y1,z0]
            grid[x0,y1,z1]=P[3].clone()+grid[x0,y1,z1]
            grid[x1,y0,z0]=P[4].clone()+grid[x1,y0,z0]
            grid[x1,y0,z1]=P[5].clone()+grid[x1,y0,z1]
            grid[x1,y1,z0]=P[6].clone()+grid[x1,y1,z0]
            grid[x1,y1,z1]=P[7].clone()+grid[x1,y1,z1]
            
        else:
            grid[x0,y0,z0]=P[0,:].clone()+grid[x0,y0,z0]
            grid[x0,y0,z1]=P[1,:].clone()+grid[x0,y0,z1]
            grid[x0,y1,z0]=P[2,:].clone()+grid[x0,y1,z0]
            grid[x0,y1,z1]=P[3,:].clone()+grid[x0,y1,z1]
            grid[x1,y0,z0]=P[4,:].clone()+grid[x1,y0,z0]
            grid[x1,y0,z1]=P[5,:].clone()+grid[x1,y0,z1]
            grid[x1,y1,z0]=P[6,:].clone()+grid[x1,y1,z0]
            grid[x1,y1,z1]=P[7,:].clone()+grid[x1,y1,z1]
            #print(dum_grid[x1[i],y1[i],z1[i]]) 
        #add the new concentrations at the gridpoints from the neuron
        #print("Dumb grid=",dum_grid)
        
        #new_grid=grid.clone()+dum_grid.clone() #this was originally done every timestep, wrong i think
        #print("we should see (x0[2],y0[2],z0[2]) grid+dumgrid as", grid[x0[2],y0[2],z0[2]],"plus",dum_grid[x0[2],y0[2],z0[2]],"=",new_grid[x0[2],y0[2],z0[2]])
        return grid
    
#Diffusion-------------------------------------------------------------------------------------

    
    #Start diffusion
    #We want input to be:
    #Concentration on the grid
    #timestep dt
    #diffusion constant
    #decay rate
    #total time for simulation
    #the rate of neurotransmitter generation



    def diffusion_step(self, grid,dx,dt,diffusion,decay): #Returns concentration of neurotransmitter
        #at each gridpoint after the next timestep

        #Create 27 point stencil (3x3x3) for discrete laplacian
        stencil=(1/(30*dx**2))*torch.tensor([[[1,3,1],[3,14,3],[1,3,1]],\
                          [[3,14,3],[14,-128,14],[3,14,3]],\
                          [[1,3,1],[3,14,3],[1,3,1]]],device='cuda',dtype=float)
        #Initialize grid size (assuming cube grid)
        grid_size=grid.size(2) #randomly picked 2
    
    
        #Create A Matrix from Joe's Writeup
        #Create Identity 3x3x3 matrix
        I=torch.eye(3,3,device='cuda',dtype=float)
        I=torch.stack([I,I,I],0)

        #Create A (3,3,3)
        A=I-0.5*diffusion*stencil*dt+0.5*dt*decay*I

        #Create Matrix B (3,3,3)
        B=I+0.5*diffusion*stencil*dt-0.5*dt*decay*I

        #To calculate vector b for Ax=b, we can use the 3d convolution
        #with B being the kernel and the grid being the input
        #We have A*u(t+1)=B*u(t)
        #Calculate b

        #Resize B for the convolution
        B=B.view(1,1,3,3,3)
        b=F.conv3d(grid.view(1,1,grid_size,grid_size,grid_size),B,padding=1)

        #We now have Ax=b
        #We could use the python linalg solver for Ax=b, but our dimensions don't line up
        #so instead we build our own conjugate gradient solver
        #and test it

        #Set max iterations for conjugate gradient
        max_iterations=20
        
        #Set threshold for conjugate gradient
        error_tol=0.0000001
        
        #Find laplacian_grid
        laplacian_grid=F.conv3d(grid.view(1,1,grid_size,grid_size,grid_size),stencil.view(1,1,3,3,3),padding=1)
        
        return CG(A,b,grid,grid_size,max_iterations,error_tol,BC,diffusion,decay,laplacian_grid,dt)




seed_everything(42)


#How to use this paracrine code------------------------------------------
#How many neurons do you have?
nnz=14000

#What is your grid size? e.g. if 32x32x32, set grid_size=32
#as of right now, we only consider uniform cube grids of grid_size**3 total points
grid_size=32

#What are the positions of the neurons? Must be values between 1 and 30 inclusive (gridpoints are 0->31 in each direction)
neuron_x=1+29*torch.rand(nnz,device='cuda',dtype=float)
neuron_y=1+29*torch.rand(nnz,device='cuda',dtype=float)
neuron_z=1+29*torch.rand(nnz,device='cuda',dtype=float)

#BOUNDARY CONDITIONS ARE JUST KEPT THE SAME FROM THE INITIAL GRIDPOINTS: CHECK THIS 
#to create our own boundary conditions, maybe we stop the padding from the 3d convolution
#then pad after every convolution takes place with the values of our boundary conditions

BC=0.005 #doesn't do anything yet


#What is your dx and dt?
dt=0.04
dx=1

#What is your diffusion and decay rate?
#0.00005,0.000004
#2e-5,3e-4,
diffusion=0.000002
decay=0.003

#What rate of neurotransmitter generated at each paracrine timestep?
generation_fraction=0.01 #this is a constant that will be multiplied by the current concentration

#What is the total time you want the simulation to run?
tmax=dt*2500


#What is the neurotransmitter concentration on grid to start?
#INITIAL CONDITION:
grid=torch.rand(grid_size,grid_size,grid_size,device='cuda',dtype=float)

#Initialize an instance of the class
model=paracrine(nnz, grid_size, neuron_x, neuron_y, neuron_z,generation_fraction,BC)


#Initialize plotting variable
plot_neuron=torch.empty(nnz,round(tmax/dt),device='cuda',dtype=float)
plot_grid=torch.empty(grid_size,grid_size,grid_size,round(tmax/dt),device='cuda',dtype=float)

#Test 1 step
#concentration_at_neuron_test=model.grid_to_neuron(grid)
#print(concentration_at_neuron_test)

#new_grid=model.production_and_neuron_to_grid(concentration_at_neuron_test,grid)
#print("new grid is",new_grid)



#Start timestep loop
for i in range(0,round(tmax/dt)):
    #Interpolate from grid to neuron to determine neurotransmitter concentration at
    #neuron locations so that we can determine how much neurotransmitter is generated
    #at the neuron locations (generation will be a function of concentration)
    
    neuron_concentrations=model.grid_to_neuron(grid) #size nnz #do every other timestep to see
    plot_neuron[:,i]=neuron_concentrations.clone() #tensor used for plotting

    
    #Now that we have neurotransmitter concentration at neuron locations, we can determine
    #how much neurotransmitter is generated and spread it back to grid
    #For now, we just add a constant
    
    grid=model.production_and_neuron_to_grid(neuron_concentrations,grid)#returns concentration at grid
    
    
    #Step the diffusion solver forward and return the concentration at gridpoints
    grid=model.diffusion_step(grid,dx,dt,diffusion,decay)
    plot_grid[:,:,:,i]=grid.clone()
    

    if (100*i/(round(tmax/dt))%1==0):
        print(100*i/(round(tmax/dt)),"%")


#Plotting
import matplotlib.pyplot as plt
#Neurons
for i in range (0,nnz):
    plt.plot(plot_neuron[i,:].to('cpu').numpy())
plt.show(block=True)

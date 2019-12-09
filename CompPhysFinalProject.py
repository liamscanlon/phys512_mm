# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:35:03 2019
Final project Nbody
@author: Liam Scanlon 
ID#: 260687833
"""




import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal
"""
1) single particle at rest should stay at rest
2) pair of particles in circular orbit should stay in orbit for a reasonable amount of time
3) Set up ~100000 particles with 
    -periodic BC 
    -nonperiodic BC 
    -Track total energy how well is it conserved?
4) start with particles on grid points with 
    (mass fluctuations should be proportinal to k^-3 so power is scale invariant)
    do this with periodic boundry conditions


"""



class Bodies:
    
    
    
    """
    initializes instance of Bodies
    m is the mass:
        if m is a scaler (eg int,double,etc.) all masses will have input mass 
        if m is an array with len(m) = nb than the mass of ith body will be m[i]
        if m = 'k^-3' than bodies are assumed to be in grid configureation, mass will be distributed in m(k) ~ k^-3 scale invartiant distrubution
    nb is the number of bodies in simulation:
        if nb is left unfilled number of bodies will default to ngrid^2 which corisponds to the number of grid points
        nb will also be set to ngrid^2 if the 'grid' option is seclected for x0 or y0
        if not overridden by condtions above nb can be set to an int wich will set the number of bodies in the simulation
    ngrid is number of gridpoints in a sidelength of the grid:
        this results in a ngrid by ngrid grid on which the simulation takes place
    fuzz set the fuzziness factor of you force/potentail ennsureing that potentail doesnt go to inf around point mass    
    x0 and y0 set the intial x and y coordinates of the bodies:
        if x0/y0 are set to 'rand' the postion of boides will be randomly distrubuted within the grid
        if x0/y0 are set to 'grid' nb will be set to ngrid^2 and a body will be placed at each grid point
            this will yeild weird and likly weird results of only x0 or y0 is set to grid
        if x0 or y0 is given an array of length nb 
    dt is the size of time step:
    bctype can be either 'wrap' or 'fill'
        'wrap' will give periodic boundry conditions
        'fill' will give no-periodic boundry condtions ( note energy of particles that leave the grid space are no longer counted with the system energy)
    potrange sets the size of the single body potential used for convolution:
        'short' : this runs faster but bodies can only feel the force of other bodies that are less than ngrid/2 grid cells away so will
                  will not work great if you care about energy consevation but if you have lots of simillarly massive bodies that the roughly
                  evenly distrubted thougout the grid space it isnt so bad since the force from the closser bodies is so much stronger than those that are father away
        'long' : this has the force act across the who grid space but is a lot slower because of the convolution step probs what you want to use
        
    
    """
    def __init__(self,m =1,nb= None,ngrid = 100,fuzz=0.03,Gn = 100, dt = 0.01,x0 ='rand',y0 = 'rand', bctype = 'wrap', potRange = 'long'  ):
        
        
        #sets nb = ngrid**2 if left as None
        if nb==None:
            nb = int(ngrid**2)
        #makes sure nb and ngird are ints
        assert isinstance(ngrid,int)==True,"ngrid must be int type"
        assert isinstance(nb,int)==True,"nb must be int type"
        assert ngrid % 2 ==0, "ngrid must be even"
        self.pars0 = {}
        
        
        
        #sets x = x0 if x0 is an array of the right length
        if isinstance(x0,np.ndarray):
            if x0.shape != (nb,):
                print('assigned x0 length doesn\'t match nb, setting x0 to random')
                x0 = 'rand'
            else: 
                self.x = x0   
        #same as above for y
        if isinstance(y0,np.ndarray):
            if y0.shape != (nb,):
                print('assigned y0 length doesn\'t match nb, setting y0 to random')
                y0 = 'rand'
            else: 
                self.y = y0
                
        #setting x and y if set to rand
        if isinstance(x0,str):
            if x0 == 'rand' or x0 == 'random':
                self.x = (ngrid-0.5)*np.random.rand(nb)
            if x0 == 'grid':
                nb = ngrid**2
                x0 = np.zeros([ngrid*ngrid])
                xrow = np.arange(ngrid)
                for i in range(ngrid):
                    x0[i*(ngrid):(i+1)*(ngrid)] = xrow.copy()
                self.x = x0 
        
        if isinstance(y0,str):
            if y0 == 'rand' or y0 == 'random':
                self.y = (ngrid-0.5)*np.random.rand(nb)          
            if y0 == 'grid':
                nb = ngrid**2
                y0 = np.zeros([ngrid*ngrid])
                yrow = np.ones(ngrid)
                for i in range(ngrid):
                    y0[i*(ngrid):(i+1)*(ngrid)] = i*yrow.copy()        
                self.y = y0
        
        
        #makes dictionary of input paramas and also sets other vaiables for instance
        self.pars0['fuzz'] = fuzz
        self.pars0['Gn'] = Gn
        self.pars0['nb'] = nb
        self.pars0['dt'] = dt 
        self.pars0['ngrid'] = ngrid        
        self.nb = nb
        self.ngrid = ngrid
        self.Gn = Gn
        self.fuzz = fuzz
        self.dt = dt
        self.bctype = bctype
        
        #sets m for varity of inputs see discription at top for more detail
        if isinstance(m,str):
            if m =="rand":
                self.m = np.random.rand(nb)
            if m == 'k^-3':
                assert (nb == ngrid**2), 'in m = \'k^-3\' mode nb must = ngrid^2 can be done automaticlly if nb is left unchanged'
                phis =2*np.pi*np.random.rand(ngrid,ngrid)
                alpha = np.cos(phis) + 1j*np.sin(phis)
                #alpha = 1
                k0 =(ngrid)/2 - 1
                mk = np.zeros([ngrid,ngrid])
                for i in range(ngrid):
                    for j in range(ngrid):
                        mk[i,j]=1/(np.sqrt((k0-i)**2+(k0-j)**2 +1 ))**3
                mx = np.abs(np.fft.ifft2(ngrid*alpha*mk))
                self.m = mx.flatten()
        else:
            self.m = np.ones(nb)*m
        self.m0 = self.m.copy()
        #intializes volcities to 0 but can be changed outside of initlization
        self.vx = np.zeros([nb])
        self.vy = 0*self.vx.copy()
        #makes sure lengths of x,y,m = nb
        if isinstance(m,np.ndarray):
            assert(len(m) == nb), "ERROR length of m does not match nb"
        assert(nb==len(self.vx) or len(self.vy)== nb or len(self.x)==len(self.y) or len(self.x) == nb), "Error postion/velocity dont match eachother or the nb"
        
        
        
        #calculates the potentail of single particle at the orgin that will be convolved later to get the graviational potentail form mass distrubution
        if potRange == 'short' :     
            print('calculating pot1')
            pot1 = np.zeros([ngrid,ngrid])
            r0 =(self.pars0['ngrid'])/2 - 1
            for i in range(self.pars0['ngrid']):
                for j in range(self.pars0['ngrid']):
                    R = np.sqrt((r0-i)**2+(r0-j)**2)
                    if R < 1:
                        pot1[i,j]=1/self.pars0['fuzz']
                    else:
                        pot1[i,j]=1/R
            print('done calc ppt1')       
            self.pot1 = pot1.copy()
            
            
            
        if potRange == 'long' :     
            print('calculating pot1')
            pot1 = np.zeros([2*ngrid,2*ngrid])
            r0 =(2*self.pars0['ngrid'])/2 - 1
            for i in range(2*self.pars0['ngrid']):
                for j in range(2*self.pars0['ngrid']):
                    R = np.sqrt((r0-i)**2+(r0-j)**2)
                    if R < 1:
                        pot1[i,j]=1/self.pars0['fuzz']
                    else:
                        pot1[i,j]=1/R
            print('done calc ppt1')       
            self.pot1 = pot1.copy()
            
            
    


    #calling instance of Bodies will output string with the masses, x and y, and vx, vy   
    def __repr__(self):
        s = '\nxs = {}  \nys = {}  \nms = {}  \nvx = {} , \nvy = {}'.format(self.x,self.y,self.m,self.vx,self.vy)
        return s 
        
        
        
        
    #calculates the mass distrubution over the grid     
    def get_rho(self):
        #gets postion of bodies snaped to nearest grid point 
        x_snap =  np.asarray(np.around(self.x),dtype = int)
        y_snap = np.asarray(np.around(self.y),dtype = int)
        x_snap = np.mod(x_snap,(self.pars0['ngrid']-1))
        y_snap = np.mod(y_snap,(self.pars0['ngrid']-1))
        rho = np.zeros([self.pars0['ngrid'],self.pars0['ngrid']])
        #print(self)
        #collects all the masses within grid cells to get the total mass within cell
        for i in range(self.pars0['nb']):
            rho[y_snap[i],x_snap[i]] = rho[y_snap[i],x_snap[i]] + self.m[i]
        if True:
            plt.clf();
            plt.imshow(rho)
            plt.colorbar()
            plt.pause(0.01)
        return rho
    #calculates the graviational potentail for each grid cell via convolution with pot1
    def calc_pot(self):
        p = signal.convolve2d(self.get_rho(),self.pars0['Gn']*self.pot1,mode = 'same',boundary = self.bctype)
        if False:
            plt.clf();
            plt.imshow(p**(1/6))
            plt.colorbar()
            #plt.plot(self.x,self.y,'.')
            plt.pause(0.01)
        return p 
    #calculates graiant of the potential to get gravitaional feild for each grid cell 
    def get_feild(self):
        kx = (1/2)*np.array([[0,0,0],[1,0,-1],[0,0,0]])
        ky = (1/2)*np.array([[0,1,0],[0,0,0],[0,-1,0]])
        pot = self.calc_pot()
        gx = signal.convolve2d(pot,kx,mode = 'same',boundary = self.bctype)
        gy = signal.convolve2d(pot,ky,mode = 'same',boundary = self.bctype)
        #print('feilds')
        #print(gx,gy)
        if False:
            plt.clf();
            plt.imshow(gx)
            plt.plot(self.x,self.y,'.')
            plt.pause(0.01)
        #returns x and y x conponents seperatly 
        return gx , gy
    
    #calculates the acceration for particles in a feild
    #this is similar to the field but now gives acceration for each particle rather than by location on grid
    def get_a(self):
        fx,fy = self.get_feild()
        x_snap =  np.asarray(np.around(self.x),dtype = int)
        y_snap = np.asarray(np.around(self.y),dtype = int)
        #x_snap = np.mod(y_snap,(self.pars0['ngrid']-1))
        #y_snap = np.mod(y_snap,(self.pars0['ngrid']-1))
        ax = np.zeros([self.pars0['nb']])
        ay = ax.copy()
        
            
            
        if self.bctype == 'fill':
            for i in range(self.pars0['nb']):
                if (y_snap[i] >= self.ngrid or x_snap[i]>= self.ngrid) or (y_snap[i]<0 or x_snap[i]<0):
                    ax[i] = 0 
                    ay[0] = 0
                else:
                    ax[i] = fx[y_snap[i],x_snap[i]]
                    ay[i] = fy[y_snap[i],x_snap[i]]
        
            
            
        if self.bctype == 'wrap':
            for i in range(self.pars0['nb']):
                ax[i] = fx[y_snap[i],x_snap[i]]
                ay[i] = fy[y_snap[i],x_snap[i]]
                        
                        
                        
        return ax,ay
    #calculates the kintic energy of the particles in the system
    def get_Ek(self):
        Ek = (1/2)*np.sum(self.m*(self.vx**2 + self.vy**2))
        return Ek
    
    #calculates the potentail energy in the system
    def get_Ep(self):
        rho = self.get_rho()
        g = signal.convolve2d(rho,self.pars0['Gn']*self.pot1,mode = 'same',boundary = self.bctype)
        #print(g.size,rho.size)
        return -np.sum(g*rho) 
    #gets the total energy if split = True than the will give the Ep Ek and Etot seperatly
    def get_E(self,split =False):
        Ek = self.get_Ek()
        Ep= self.get_Ep()
        E = Ep + Ek
        if split == False:
            return E
        if split == True:
            return Ep , Ek , E
    #its in the name! :)
    def get_momentum(self):
        px = np.sum(self.m*self.vx)
        py = np.sum(self.m*self.vy)
        ptot = np.sqrt(px**2 + py**2)
        return px,py,ptot
        
    #advances time using leapfrog method
    def advance_time(self,nt):
        #print(self.get_E(split=True))
        #print(self)
        for i in range(nt):
            print('iteration # = ' + str(i))
            ax,ay = self.get_a() 
            self.x = self.x + self.pars0['dt']*self.vx + (1/2)*ax*self.pars0['dt']**2 
            self.y =  self.y + self.pars0['dt']*self.vy + (1/2)*ay*self.pars0['dt']**2
            #for wrap bc bodies that go past boundry loop back around
            if self.bctype == 'wrap':
                self.x = np.mod(self.x,(self.pars0['ngrid']-1))
                self.y = np.mod(self.y,(self.pars0['ngrid']-1))
            #this is kinda a lazy hack to make bodies that leave the system to not interact with other bodies
            if self.bctype == 'fill':
                xout = np.logical_or((self.x>self.ngrid),self.x<0)
                yout = np.logical_or((self.y>self.ngrid),self.y<0)
                out = np.logical_or(xout,yout)
                self.x[out] = 0
                self.y[out] = 0 
                self.m[out] = 0 
                
            if False:
                plt.clf();
                plt.axis([0,self.pars0['ngrid'],0,self.pars0['ngrid']])
                plt.plot(self.x,self.y,'.')
                plt.pause(1) 
            axnew , aynew = self.get_a() 
            self.vx =self.vx + (1/2)*(ax + axnew)*self.pars0['dt']**2
            self.vy = self.vy + (1/2)*(ay + aynew)*self.pars0['dt']**2
            


"""
Lonely sationary body should stay where it is!
**for full effect please listen to: "One (Is The Loneliest Number)"**
"""    
if False:
    #setup single body simulation
    #x0 and y0 
    b1 = Bodies(nb=1,Gn = 1000, potRange='long')
    #get starting energy    
    E0 = b1.get_E()
    #advance time
    b1.advance_time(25)
    #get energy after time has advanced
    E1 = b1.get_E()
    #prints change in energy
    print('change in energy for stationary lonely body =  {} '.format(E1-E0))
"""    
It works!!! yay 
stays still and energy is conserved    
"""    
    
"""
since the the boundry condtions can screw with the conservation of momentum 
lets set it moving towards boundry with periodic BCs and see if we still obey 
Newtons laws
"""
if False:
    b1.x[0] = 0
    b1.vx[0] = -80
    E0 = b1.get_E()
    b1.advance_time(25)
    E1 = b1.get_E()
    print('change in energy for lonely  body moving through periodic boundry  = {}'.format(E1-E0))

"""
Now with the non periodic BCs
"""
if False:
    #setup single body simulation 
    #x0 and y0 
    b1 = Bodies(nb=1,Gn = 100,bctype = 'fill')
    #get starting energy    
    E0 = b1.get_E()
    #advance time
    b1.advance_time(25)
    #get energy after time has advanced
    E1 = b1.get_E()
    #prints change in energy
    print('change in energy for stationary lonely body =  {} '.format(E1-E0))


    b1.x[0] = 0
    b1.vx[0] = -80
    E0 = b1.get_E()
    print("starting energy is {}".format(E0) )
    b1.advance_time(25)
    E1 = b1.get_E()
    print('change in energy for lonely  body moving past  boundry  = {}'.format(E1-E0))
    print('Here we can see that the enery goes to zero because the body left the system')
"""
works like this too!! 

"""



"""
2 bodies in orbit
condtion for circular orbit:
    a = v^2/r
    v = sqrt(r*a)

"""
if False:
    x0_2 = np.array([35,65])
    y0_2 = np.array([50,50])
    m0 = 5*np.array([1,1])
    b2 = Bodies(m = m0,nb=2,ngrid = 100,Gn = 6000000, x0 =x0_2,y0=y0_2,dt = 0.001,bctype = 'fill')
    #b2.advance_time(60)
    p1 =b2.pot1
    #plt.figure();plt.imshow(p1)
    
    
    
    
    #b2.vx[1] = np.sqrt(b2.Gn*(np.sum(b2.m))/np.sqrt((b2.x[0]-b2.x[1])**2+(b2.y[0]-b2.y[1])**2))
    ax , ay = b2.get_a()
    R0 = np.sqrt((b2.x[0]-b2.x[1])**2+(b2.y[0]-b2.y[1])**2)
    dv = 1*np.sqrt(np.abs(ax[1])*R0/2)
    b2.vy[0] = dv/2
    b2.vy[1] = -dv/2
    b2.m[0] = b2.m[1]
    Eb2_0 = b2.get_E(split = True)
    b2.advance_time(100)
    R1 = np.sqrt((b2.x[0]-b2.x[1])**2+(b2.y[0]-b2.y[1])**2)
    Eb2_1 = b2.get_E(split = True) 
    #print('change in energy for 2 bodies in orbit = {}'.format(Eb2_1-Eb2_0))
    print(Eb2_0)
    print(Eb2_1)
    print(R0)
    print(R1)
"""
this one is a little more tricky 
it is diffcult to keep things in orbit because the acceraertion felt by a body is determined by Ug at 
of a grid point. since we are working with spherical orbits the potentail of the circular path 
is not perfect since you cant draw a circle out of squares so once a body is snaped to 
a grid cell that is father away from the other body than the obit gets bigger and there isn't 
engough pull from the other body to counteract the volocity of the body be contained.
-the volocities of the bodies still match eachother 
-energy is still pretty well preserved given that Ep is also calculated using the pretty chunky grid 
potentails
"""






"""
Lots of bodies
"""
if False:    
     
    bp = Bodies(m = 'rand',nb=200000,ngrid = 100,Gn = 100, x0 ='rand',y0='rand',dt = 0.01,bctype = 'wrap', fuzz = 0.05,potRange='long')
    Ep_0 = bp.get_E()
    bp.advance_time(25)
    Ep_1 = bp.get_E()
    dE = Ep_1-Ep_0
    print('change in energy = {} '.format(dE))
    print('percent change = {}%'.format(100*dE/Ep_0))
    p = bp.get_momentum()
    print('change in p = {}'.format(p[2]))
"""
This one works pretty well though the energy does not look like it is tha well conserved (change is about ~5% to 10%) but again I think 
this is more due to the low res Ep calculator  
    I think if you fine tuned the values of dt, fuzz and Gn you might be able to get some more relistic results
    also increasing the grid size would fix a lot of these issues but would also slow down the simulation a lot
"""    
    
    
if False:    
    bnp = Bodies(m = 'rand',nb=200000,ngrid = 100,Gn = 100, x0 ='rand',y0='rand',dt = 0.01,bctype = 'fill',fuzz = 0.05,potRange= 'long'   )
    Enp_0 = bnp.get_E()
    bnp.advance_time(25)
    Enp_1 = bnp.get_E()
    dE = Enp_1-Enp_0
    print('change in energy = {} '.format(dE))
    print('percent change = {}%'.format(100*dE/Enp_0))
"""
this one does not conserve energy but that is to be expected since bodies leave the system
bodies clump together then pass through eachother and then clump back together due to "fuzz factor" and once things clump 
together engough the fuzzyness of the bodies means that gravitaional force in no longer particularlly
conservative.
kinda looks like a clump of space dust??
    hard to tell since astrophysics to me is 60% black magic
"""    
    
    
    

"""
now with k^-3 mass distrubution and grid placement in periodic BCs
"""

if True:
    bk = Bodies(m='k^-3',ngrid= 100, Gn= 86900, x0 = 'grid',y0= 'grid', fuzz = 0.04 ,potRange='short',dt = 0.008)
    #bk.vx = np.random.randn(len(bk.vx))
    #bk.vy = np.random.randn(len(bk.vx))
    p0 = bk.get_momentum()
    Ek_0 = bk.get_E(split = True)
    bk.advance_time(20)
    Ek_1 = bk.get_E(split = True)
    dE = Ek_1[2]-Ek_0[2]
    p1 = bk.get_momentum()
    print('change in energy = {} '.format(dE))
    print('percent change = {}%'.format(100*dE/Ek_0[2]))
    print('change in momentum = {} '.format(p1[2] - p0[2]))

"""
this one is actually kinda neat tho it would have been nice if we had been told what we should be looking for here. unlike when we had randomly
distrbuted masses and positions this looks more spacey. structures seem to form and stay fairly stable at larger scales and some whispy bits coming off with some instresting 
dynamics within the structures. energy isnt conserved perfectly but as explained throught this is at least in part due to how we get the Ep 
values. 
"""

"""
Notes:
    So as not to brick your computer I have all the tests "behind" if statments so if you want to test it out just set that one to true
    
    I did this in 2D but this code could easily be extended to 3D by adding a self.x self.vz and making pot1 NxNxN but since my code is 
    not very speedy this would be slow. why stop at flat earth now we are all about flat universe.
    
    
    
    Right now I have it set so it displays the dencity after every time step but you can also get it to show you particle postions, gravitional
    feild in x or y direction gravitional potentail etc with the if staments in the Bodies class. 
    
    
    there are quite a few things I could have improved here which I didnt, due to time constraints. 
    -Writing my own convolution function that was better optimized for this project would likly the best way to speed up the code since the 
     built in scipy one is a little slow for this kind of aplication it wouldnt be that hard with a few fft2's and ifft2's from numpy to do 
     but I didnt relize this was the bottle neck of the code untill far enough in the debugging stage that I didnt have time to implement and 
     debug it.
    -having the program do some of the loops in C would also help particularly in the get_rho and get_a functions
    -I think that my boundry conditions could use a bit more debugging than I have time for
    -I had this grand idea of making the pot1 by doing one quadrant then using sysmtry to stick rotated versions together
     into one bigger array but I ran out of brain power some point and rather than hitting my head on a wall I just went with the way I did that 
     was a little slower (particularlly setting up for large grid sizes) but still not the bottle neck by a long shot
    -had I done the things mensioned above a larger gridsize would have been a lot faster making a lot of the energy conservation problems I 
     was having go away.
     
     To expand on what I was trying to say about the potentail energy calculations:
         Ep is calculated by adding up the product of the gravitaional potentail and the mass within each grid cell. regardless of 
         how where in the grid cell a body sits it is assumed to have the to potentail energy of a body at the center of the cell 
         this introduces a fair amount of error when there are few bodies per grid point and when bodies are close enough together that the 
         true 1/r potentail varries a lot within a grid cell. the easy solution to this is to have a higher res grid but for the reasons above
         that would take a long time to this is what you get :S. 
     
     
     
     
     
     Also I spent way to long trying to save these as .gif or .mp4 files but this ended up being such a pain that I gave up so 
     Im sorry but you will probs have to run the code yourself. I tried to make it easy for you I really did :(

    In closing please be kind I really need to pass this course. 



"""













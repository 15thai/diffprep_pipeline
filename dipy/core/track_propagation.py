import os
import numpy as np


class FACT_Delta():
    ''' Generates tracks with termination criteria defined by a
    delta function [1]_ and it has similarities with FACT algorithm [2]_.

    Can be used with any reconstruction method as DTI,DSI,QBI,GQI which can
    calculate an orientation distribution function and find the local peaks of
    that function. For example a single tensor model can give you only
    one peak a dual tensor model 2 peaks and quantitative anisotropy
    method as used in GQI can give you 3,4,5 or even more peaks.    
    
    The parameters of the delta function are checking thresholds for the
    direction propagation magnitude and the angle of propagation.

    A specific number of seeds is defined randomly and then the tracks
    are generated for that seed if the delta function returns true.

    Trilinear interpolation is being used for defining the weights of
    the propagation.

    References
    ----------

    .. [1] Yeh. et al. Generalized Q-Sampling Imaging, TMI 2010.

    .. [2] Mori et al. Three-dimensional tracking of axonal projections
    in the brain by magnetic resonance imaging. Ann. Neurol. 1999.
    

    '''

    def __init__(self,qa,ind,seeds_no=1000,odf_vertices=None,qa_thr=0.0239,step_sz=0.5,ang_thr=60.):
        '''
        Parameters
        ----------

        qa: array, shape(x,y,z,Np), magnitude of the peak (QA) or
        shape(x,y,z) a scalar volume like FA.

        ind: array, shape(x,y,z,Np), indices of orientations of the QA
        peaks found at odf_vertices used in QA or, shape(x,y,z), ind

        seeds_no: number of random seeds

        odf_vertices: sphere points which define a discrete
        representation of orientations for the peaks, the same for all voxels

        qa_thr: float, threshold for QA(typical 0.023)  or FA(typical 0.2) 
        step_sz: float, propagation step

        ang_thr: float, if turning angle is smaller than this threshold
        then tracking stops.        

        Returns
        -------

        tracks: sequence of arrays

        '''

        if len(qa.shape)==3:

            qa.shape=qa.shape+(1,)
            ind.shape=ind.shape+(1,)

        self.Np=qa.shape[-1]

        x,y,z,g=qa.shape
        tlist=[]
        

        if odf_vertices==None:

            eds=np.load(os.path.join(os.path.dirname(__file__),'matrices',\
                        'evenly_distributed_sphere_362.npz'))
            odf_vertices=eds['vertices']

        for i in range(seeds_no):

            rx=(x-1)*np.random.rand()
            ry=(y-1)*np.random.rand()
            rz=(z-1)*np.random.rand()
            
            seed=np.array([rx,ry,rz])
            track=self.propagation(seed,qa,ind,odf_vertices,qa_thr,ang_thr,step_sz)

            if track == None:
                pass
            else:
                tlist.append(track)

        self.tracks=tlist
            


    def trilinear_interpolation(self,X):

        Xf=np.floor(X)        
        #d holds the distance from the (floor) corner of the voxel
        d=X-Xf
        #nd holds the distance from the opposite corner
        nd = 1-d
        #filling the weights
        W=np.array([[ nd[0] * nd[1] * nd[2] ],
                    [  d[0] * nd[1] * nd[2] ],
                    [ nd[0] *  d[1] * nd[2] ],
                    [ nd[0] * nd[1] *  d[2] ],
                    [  d[0] *  d[1] * nd[2] ],
                    [ nd[0] *  d[1] *  d[2] ],
                    [  d[0] * nd[1] *  d[2] ],
                    [  d[0] *  d[1] *  d[2] ]])

        IN=np.array([[ Xf[0]  , Xf[1]  , Xf[2] ],
                    [ Xf[0]+1 , Xf[1]  , Xf[2] ],
                    [ Xf[0]   , Xf[1]+1, Xf[2] ],
                    [ Xf[0]   , Xf[1]  , Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]+1, Xf[2] ],
                    [ Xf[0]   , Xf[1]+1, Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]  , Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]+1, Xf[2]+1 ]])


        return W,IN.astype(np.int)

    def nearest_direction(self,dx,qa,ind,odf_vertices,qa_thr=0.0245,ang_thr=60.):

        ''' Give the nearest direction to a point

        Parameters
        ----------
        
        dx: array, shape(3,), as float, moving direction of the current
        tracking

        qa: array, shape(Np,), float, quantitative anisotropy matrix,
        where Np the number of peaks, found using self.Np

        ind: array, shape(Np,), float, index of the track orientation

        odf_vertices: array, shape(N,3), float, odf sampling directions

        qa_thr: float, threshold for QA, we want everything higher than
        this threshold 

        ang_thr: float, theshold, we only select fiber orientation with
        this range 


        Returns
        --------

        delta: bool, delta funtion, if 1 we give it weighting if it is 0
        we don't give any weighting

        direction: array, shape(3,), the fiber orientation to be
        consider in the interpolation



        '''

        max_dot=0

        max_doti=0

        angl = np.cos((np.pi*ang_thr)/180.) 

        if qa[0] <= qa_thr:

            return False, np.array([0,0,0])


        for i in range(self.Np):

            if qa[i]<= qa_thr:

                break

            curr_dot = np.abs(np.dot(dx, odf_vertices[ind[i]]))

            if curr_dot > max_dot:

                max_dot = curr_dot

                max_doti = i


        if max_dot < angl :

            return False, np.array([0,0,0])


        if np.dot(dx,odf_vertices[ind[max_doti]]) < 0:

            return True, - odf_vertices[ind[max_doti]]

        else:

            return True,   odf_vertices[ind[max_doti]]


        
    def propagation_direction(self,point,dx,qa,ind,odf_vertices,qa_thr,ang_thr):
        ''' Find where you are moving next

        '''

        total_w = 0 # total weighting
        new_direction = np.array([0,0,0])
        w,index=self.trilinear_interpolation(point)

        #check if you are outside of the volume
        for i in range(3):
            if index[7][i] >= qa.shape[i] or index[0][i] < 0:
                return False, np.array([0,0,0])


        #calculate qa & ind of each of the 8 corners
        for m in range(8):

            x,y,z = index[m]
            qa_tmp = qa[x,y,z]
            ind_tmp = ind[x,y,z]
            delta,direction = self.nearest_direction(dx,qa_tmp,ind_tmp,odf_vertices,qa_thr,ang_thr)
            #print delta, direction
            if not delta:
                continue

            total_w += w[m]
            new_direction = new_direction +  w[m][0]*direction

        if total_w < .5: # termination criteria
            return False, np.array([0,0,0])

        return True, new_direction/np.sqrt(np.sum(new_direction**2))
    
    def initial_direction(self,seed,qa,ind,odf_vertices,qa_thr):
        ''' First direction that we get from a seeding point

        '''

        #very tricky/cool addition/flooring that helps create a valid
        #neighborhood (grid) for the trilinear interpolation to run smoothly
        seed+=0.5
        point=np.floor(seed)
        x,y,z = point
        qa_tmp=qa[x,y,z,0]#maximum qa
        ind_tmp=ind[x,y,z,0]#corresponing orientation indices for max qa

        if qa_tmp < qa_thr:
            return False, np.array([0,0,0])
        else:
            return True, odf_vertices[ind_tmp]


    def propagation(self,seed,qa,ind,odf_vertices,qa_thr,ang_thr,step_sz):
        '''

        Parameters
        ----------

        seed: array, shape(3,), point where the tracking starts
        
        qa: array, shape(Np,), float, quantitative anisotropy matrix,
        where Np the number of peaks, found using self.Np

        ind: array, shape(Np,), float, index of the track orientation
        
                
        Returns
        -------
        d: bool, delta function result
        
        idirection: array, shape(3,), index of the direction of the propagation

        '''
        #d is the delta function 
        d,idirection=self.initial_direction(seed,qa,ind,odf_vertices,qa_thr)
        #print d
        if not d:
            return None

        dx = idirection
        point = seed
        track = []
        track.append(point)

        #track towards one direction 
        while d:

            d,dx = self.propagation_direction(point,dx,qa,ind,\
                                                  odf_vertices,qa_thr,ang_thr)
            if not d:
                break

            point = point + step_sz*dx
            track.append(point)

        d = True
        dx = - idirection
        point = seed

        #track towards the opposite direction
        while d:
            d,dx = self.propagation_direction(point,dx,qa,ind,\
                                         odf_vertices,qa_thr,ang_thr)
            if not d:
                break

            point = point + step_sz*dx
            track.insert(0,point)

        return np.array(track)

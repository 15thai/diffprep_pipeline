#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from .fused_types cimport floating, bool
import numpy.linalg as npl


cdef extern from "dpy_math.h" nogil:
    double floor(double)
    double ceil(double)
    double sqrt(double)
    double cos(double)


def quadratic_transform(floating[:, :, :] volume, int[:] shape, double [:] quad_params, double[:,:] matrix, double[:, :] sampling_grid2world, double [:,:] codomain_grid2world, int phase_id, int do_cubic, interp = 'linear'):
    if interp == 'linear':
        do_quad_int = 0
    else:
        do_quad_int = 1
    

    codomain_grid2world_inv= npl.inv(codomain_grid2world)
    cdef double [:,:] codomain_grid2world_inv_view = codomain_grid2world_inv
    cdef:
        cnp.npy_intp nY = shape[1]
        cnp.npy_intp nX = shape[0]
        cnp.npy_intp nslices = shape[2]
        cdef double p[3]
        cnp.npy_intp i, j, k, ii, jj, kk
        double x,y,z, xp, yp, zp, new_phase_coord, total_change, ypp
        double ip,jp,kp
        floating [:, :, :] out = np.zeros(( nX, nY,nslices),
                                        dtype=np.asarray(volume).dtype)
        int inside
        cnp.npy_intp do_quad = do_quad_int
    
    total_change = 0.

	
    with nogil:
        for k in range(nslices):
            p[0] = 0
            p[1] = 0
            p[2] = 0
            for i in range(nX):
                for j in range(nY):
		    #     OutputPointType p = m_Matrix * point + m_Translation;
                    x= sampling_grid2world[0,0] *i + sampling_grid2world[0,1] *j +  sampling_grid2world[0,2] *k + sampling_grid2world[0,3]
                    y= sampling_grid2world[1,0] *i + sampling_grid2world[1,1] *j +  sampling_grid2world[1,2] *k + sampling_grid2world[1,3]
                    z= sampling_grid2world[2,0] *i + sampling_grid2world[2,1] *j +  sampling_grid2world[2,2] *k + sampling_grid2world[2,3]

                    xp = matrix[0,0]*x + matrix[0,1]*y + matrix[0,2]*z + quad_params[0]
                    yp = matrix[1,0]*x + matrix[1,1]*y + matrix[1,2]*z + quad_params[1]
                    zp = matrix[2,0]*x + matrix[2,1]*y + matrix[2,2]*z + quad_params[2]

                    new_phase_coord = (quad_params[6]* xp + quad_params[7] * yp + quad_params[8] * zp + \
                                quad_params[9] * xp * yp + quad_params[10] * xp * zp +  quad_params[11] * yp * zp + \
                                 quad_params[12] * (xp * xp - yp * yp) + quad_params[13] * (2 * zp * zp - xp * xp - yp * yp))

                    if do_cubic==1:
                        total_change = (quad_params[14] * xp * yp * zp + \
                                         quad_params[15] * zp * (xp * xp - yp * yp) + \
                                         quad_params[16] * xp * (4 * zp * zp - xp * xp - yp * yp) + \
                                         quad_params[17] * yp * (4 * zp * zp - xp * xp - yp * yp) + quad_params[18] * xp * (xp * xp - 3 * yp * yp) + \
                                         quad_params[19] * yp * (3 * xp * xp - yp * yp) + quad_params[20] * zp * (2 * zp * zp - 3 * xp * xp - 3 * yp * yp))
                    
                    ypp = total_change + new_phase_coord
                    p[0] = xp
                    p[1] = yp		
                    p[2] = zp
                    p[phase_id] = ypp
                    ip = codomain_grid2world_inv_view[0,0]*p[0] + codomain_grid2world_inv_view[0,1]*p[1] + codomain_grid2world_inv_view[0,2]*p[2] + codomain_grid2world_inv_view[0,3]
                    jp = codomain_grid2world_inv_view[1,0]*p[0] + codomain_grid2world_inv_view[1,1]*p[1] + codomain_grid2world_inv_view[1,2]*p[2] + codomain_grid2world_inv_view[1,3]
                    kp = codomain_grid2world_inv_view[2,0]*p[0] + codomain_grid2world_inv_view[2,1]*p[1] + codomain_grid2world_inv_view[2,2]*p[2] + codomain_grid2world_inv_view[2,3]

                    if do_quad:
                        inside = _cubic_interpolate3d[floating](volume, ip, jp, kp, &out[i,j,k])
                    else:
                        inside = _linear_interpolate[floating](volume, ip, jp, kp, &out[i,j,k])


    return np.asarray(out)


cdef inline int _cubic_interpolate3d(floating [:,:,:]volume, double ip, double jp, double kp, floating *out) nogil:

    cdef:    
        double currX=ip
        double currY=jp
        double currZ=kp

        double Z_vector[4]
        double Y_vector[4]
        double X_vector[4]
        cnp.npy_intp startX,startY,startZ
        double X,Y,Z
        double mult=1./6
    

        cnp.npy_intp nx = volume.shape[0]
        cnp.npy_intp ny = volume.shape[1]
        cnp.npy_intp nz = volume.shape[2]


        int S0=nx
        int S0S1=nx*ny
        
        double t_m1_m1,t_m1_0,t_m1_1,t_m1_2
        double t_0_m1,t_0_0,t_0_1,t_0_2
        double t_1_m1,t_1_0,t_1_1,t_1_2
        double t_2_m1,t_2_0,t_2_1,t_2_2
        double u_m1,u_0,u_1,u_2

        int tmp0, tmp1, tmp2,tmp3
    #with gil:
    #    print(ip,jp,kp, nx, ny, nz)
    if (ip < 0 or ip > nx-2 or jp < 0 or jp > ny-2 or kp < 0 or kp > nz-2):

        out[0] = 0
        return 0


    startX = <int>floor(ip)
    startY = <int>floor(jp)
    startZ = <int>floor(kp)
    X= currX - startX
    Y= currY - startY
    Z= currZ - startZ

    Z_vector[0] = mult* -Z*(Z-1)*(Z-2)
    Z_vector[1] = 0.5*  (Z+1)*(Z-1)*(Z-2)
    Z_vector[2] = -0.5* Z*(Z+1)*(Z-2)
    Z_vector[3] = mult* Z*(Z-1)*(Z+1)


    Y_vector[0] = mult* -Y*(Y-1)*(Y-2)
    Y_vector[1] = 0.5*  (Y+1)*(Y-1)*(Y-2)
    Y_vector[2] = -0.5* Y*(Y+1)*(Y-2)
    Y_vector[3] = mult* Y*(Y-1)*(Y+1)


    X_vector[0] = mult* -X*(X-1)*(X-2)
    X_vector[1] = 0.5*  (X+1)*(X-1)*(X-2)
    X_vector[2] = -0.5* X*(X+1)*(X-2)
    X_vector[3] = mult* X*(X-1)*(X+1)

    #if (startX-2 < 0) or  (startY-2<0)  or (startZ-2 <0) or (startX + 1 > nx) or (startY + 1>ny) or (startZ + 1>nz):
    #    with gil:
    #        print(startX, startY, startZ)
    t_m1_m1 = Z_vector[0]*volume[startX-1, startY-1, startZ-1] + \
              Z_vector[1]*volume[startX-1, startY-1, startZ] + \
              Z_vector[2]*volume[startX-1, startY-1, startZ+1] + \
              Z_vector[3]*volume[startX-1, startY-1, startZ+2]

    t_m1_0 = Z_vector[0]*volume[startX-1, startY, startZ-1] + \
             Z_vector[1]*volume[startX-1, startY, startZ ] + \
             Z_vector[2]*volume[startX-1, startY, startZ+1] + \
             Z_vector[3]*volume[startX-1, startY, startZ+2]

    t_m1_1 = Z_vector[0]*volume[startX-1, startY+1, startZ-1]  + \
              Z_vector[1]*volume[startX-1, startY+1, startZ]  + \
              Z_vector[2]*volume[startX-1, startY+1, startZ+1]  + \
              Z_vector[3]*volume[startX-1, startY+1, startZ+2]

    t_m1_2 = Z_vector[0]*volume[startX-1, startY+2, startZ-1]  + \
             Z_vector[1]*volume[startX-1, startY+2, startZ]  + \
             Z_vector[2]*volume[startX-1, startY+2, startZ+1]  + \
             Z_vector[3]*volume[startX-1, startY+2, startZ+2]

    t_0_m1 = Z_vector[0]*volume[startX, startY-1, startZ-1]  + \
             Z_vector[1]*volume[startX, startY-1, startZ]  + \
             Z_vector[2]*volume[startX, startY-1, startZ+1]  + \
             Z_vector[3]*volume[startX, startY-1, startZ+2]

    t_0_0 = Z_vector[0]*volume[startX, startY, startZ-1]  + \
             Z_vector[1]*volume[startX, startY, startZ]  + \
             Z_vector[2]*volume[startX, startY, startZ+1]  + \
             Z_vector[3]*volume[startX, startY, startZ+2]

    t_0_1 = Z_vector[0]*volume[startX, startY+1, startZ-1]  + \
             Z_vector[1]*volume[startX, startY+1, startZ]  + \
             Z_vector[2]*volume[startX, startY+1, startZ+1]  + \
             Z_vector[3]*volume[startX, startY+1, startZ+2]

    t_0_2 = Z_vector[0]*volume[startX+1, startY+2, startZ-1]  + \
             Z_vector[1]*volume[startX+1, startY+2, startZ]  + \
             Z_vector[2]*volume[startX+1, startY+2, startZ+1]  + \
             Z_vector[3]*volume[startX+1, startY+2, startZ+2]

    t_1_m1 = Z_vector[0]*volume[startX+1, startY-1, startZ-1]  + \
             Z_vector[1]*volume[startX+1, startY-1, startZ]  + \
             Z_vector[2]*volume[startX+1, startY-1, startZ+1]  + \
             Z_vector[3]*volume[startX+1, startY-1, startZ+2]

    t_1_0 = Z_vector[0]*volume[startX+1, startY, startZ-1]  + \
             Z_vector[1]*volume[startX+1, startY, startZ]  + \
             Z_vector[2]*volume[startX+1, startY, startZ+1]  + \
             Z_vector[3]*volume[startX+1, startY, startZ+2]

    t_1_1 = Z_vector[0]*volume[startX+1, startY+1, startZ-1]  + \
             Z_vector[1]*volume[startX+1, startY+1, startZ]  + \
             Z_vector[2]*volume[startX+1, startY+1, startZ+1]  + \
             Z_vector[3]*volume[startX+1, startY+1, startZ+2]

    t_1_2 = Z_vector[0]*volume[startX+1, startY+2, startZ-1]  + \
             Z_vector[1]*volume[startX+1, startY+2, startZ]  + \
             Z_vector[2]*volume[startX+1, startY+2, startZ+1]  + \
             Z_vector[3]*volume[startX+1, startY+2, startZ+2]

    t_2_m1 = Z_vector[0]*volume[startX+2, startY-1, startZ-1]  + \
             Z_vector[1]*volume[startX+2, startY-1, startZ]  + \
             Z_vector[2]*volume[startX+2, startY-1, startZ+1]  + \
             Z_vector[3]*volume[startX+2, startY-1, startZ+2]

    t_2_0 = Z_vector[0]*volume[startX+2, startY, startZ-1]  + \
             Z_vector[1]*volume[startX+2, startY, startZ]  + \
             Z_vector[2]*volume[startX+2, startY, startZ+1]  + \
             Z_vector[3]*volume[startX+2, startY, startZ+2]

    t_2_1 = Z_vector[0]*volume[startX+2, startY+1, startZ-1]  + \
             Z_vector[1]*volume[startX+2, startY+1, startZ]  + \
             Z_vector[2]*volume[startX+2, startY+1, startZ+1]  + \
             Z_vector[3]*volume[startX+2, startY+1, startZ+2]

    t_2_2 = Z_vector[0]*volume[startX+2, startY+2, startZ-1]  + \
             Z_vector[1]*volume[startX+2, startY+2, startZ]  + \
             Z_vector[2]*volume[startX+2, startY+2, startZ+1]  + \
             Z_vector[3]*volume[startX+2, startY+2, startZ+2]

    u_m1=    Y_vector[0]*t_m1_m1 + Y_vector[1]*t_m1_0 + Y_vector[2]*t_m1_1 + Y_vector[3]*t_m1_2
    u_0=     Y_vector[0]*t_0_m1 + Y_vector[1]*t_0_0 +Y_vector[2]*t_0_1 +Y_vector[3]*t_0_2
    u_1=     Y_vector[0]*t_1_m1 + Y_vector[1]*t_1_0 +Y_vector[2]*t_1_1 +Y_vector[3]*t_1_2
    u_2=     Y_vector[0]*t_2_m1 + Y_vector[1]*t_2_0 +Y_vector[2]*t_2_1 +Y_vector[3]*t_2_2

    out[0]=X_vector[0]*u_m1 + X_vector[1]*u_0+X_vector[2]*u_1 + X_vector[3]*u_2
    return 1


cdef int _linear_interpolate(floating[:, :, :] volume, double ip, double jp, double kp,   floating *out)  nogil:

    cdef:
        cnp.npy_intp floor_x,floor_y,floor_z,ceil_x,ceil_y,ceil_z
        double xd, yd,zd
        double i1a,i1b,i1
        double i2a,i2b,i2
        double j1a,j1b,j1
        double j2a,j2b,j2
        double w1,w2


        cnp.npy_intp nx = volume.shape[0]
        cnp.npy_intp ny = volume.shape[1]
        cnp.npy_intp nz = volume.shape[2]

    if (ip < 0 or ip > nx or jp < 0 or jp > ny or kp < 0 or kp > nz):
        out[0] = 0
        return 0

    floor_x = <int>floor(ip)
    floor_y = <int>floor(jp)
    floor_z = <int>floor(kp)
    ceil_x = <int>ceil(ip)
    ceil_y = <int>ceil(jp)
    ceil_z = <int>ceil(kp)

    xd= ip-floor_x
    yd= jp-floor_y
    zd= kp-floor_z

    i1a=volume[floor_x,floor_y,floor_z]
    i1b=volume[floor_x,floor_y,ceil_z]
    i1= i1a*(1-zd) + i1b*zd

    i2a=volume[floor_x,ceil_y,floor_z]
    i2b=volume[floor_x,ceil_y,ceil_z]
    i2= i2a*(1-zd) + i2b*zd


    j1a=volume[ceil_x,floor_y,floor_z]
    j1b=volume[ceil_x,floor_y,ceil_z]
    j1= j1a*(1-zd) + j1b*zd

    j2a=volume[ceil_x,ceil_y,floor_z]
    j2b=volume[ceil_x,ceil_y,ceil_z]
    j2= j2a*(1-zd) + j2b*zd

    w1= i1*(1-yd) + i2*yd
    w2= j1*(1-yd) + j2*yd

    out[0]= w1*(1-xd) + w2*xd

    return 1

# -*- coding: utf-8 -*-

cimport numpy as cnp
from libc.math cimport isfinite, NAN
import numpy as np
from numpy import linalg
import math
from libc.stdlib cimport malloc, free
from libc.stdlib cimport qsort

cdef double * Hbuffer = NULL
cdef int * Hindex = NULL
cdef int Hsize = 0
cdef int gNT = 0

cdef inline double vabs(double x) nogil:
    if(x>=0):
        return x
    return -x

cdef void insert(double * H, int index, double value) nogil:
    cdef int i
    for i in range(index):
        H[i] = H[i+1]
    H[index] = value

cdef void int_insert(int * H, int index, int value) nogil:
    cdef int i
    for i in range(index):
        H[i] = H[i+1]
    H[index] = value

cdef double cvtdot(double * X1, double * X2, int N, int NT) nogil:
    return ptddot(N,X1,1,X2,1,NT)

cdef double tddot(int N, double * X1, int incx, double * X2, int incy) nogil:
    return ptddot(N,X1,incx,X2,incy,gNT)

cdef int compare_ints(const void * x, const void * y) nogil:
    if(((<int *>x)[0]) < ((<int *>y)[0])):
        return -1
    elif(((<int *>x)[0]) > ((<int *>y)[0])):
        return 1
    return 0


cdef int compare_doubles(const void * x, const void * y) nogil:
    if(not isfinite((<double *>x)[0])):
        return 1
    if(not isfinite((<double *>y)[0])):
        return -1
    if(vabs((<double *>x)[0]) < vabs((<double *>y)[0])):
        return -1
    elif(vabs((<double *>x)[0]) > vabs((<double *>y)[0])):
        return 1
    return 0

cdef double oldptddot(int N, double * X1, int incx, double * X2, int incy, int NT) nogil:
    global Hbuffer, Hsize
#on ajuste la taille du buffer si necessaire
    if(N > Hsize):
        if(Hbuffer != NULL):
            free(Hbuffer)
        Hbuffer = <double *> malloc(sizeof(double)*N)
    cdef int i
    cdef double dot = 0.0
    for i in range(N):
        Hbuffer[i] = X1[i*incx]*X2[i*incy]

    qsort(Hbuffer, N, sizeof(double), compare_doubles)

    for i in range(N-NT):
        dot += Hbuffer[i]

    return dot

cdef double ptddot(int N, double * X1, int incx, double * X2, int incy, int NT) nogil:
    global Hbuffer, Hsize, Hindex
#on ajuste la taille du buffer si necessaire
    if(NT > Hsize):
        if(Hbuffer != NULL):
            free(Hbuffer)
        Hbuffer = <double *> malloc(sizeof(double)*NT)

    if(NT > Hsize):
        if(Hindex != NULL):
            free(Hindex)
        Hindex = <int *> malloc(sizeof(int)*NT)
#calcul du produit scalaire
    cdef double dot = 0.0
    cdef int i
    cdef double val = 0.0
    cdef int index
    cdef int offset
    cdef int end = NT-1
    cdef int cpNT = NT

    for i in range(NT):
        Hbuffer[i] = 0.0

    for i in range(N):
        if( (not isfinite(X1[i*incx]*X2[i*incy])) ):
            if(NT==0):
                return NAN
            insert(Hbuffer,end,0.0)
            int_insert(Hindex,end,i)
            end -= 1
            NT -= 1
            continue
        val = (X1[i*incx])*(X2[i*incy])
        if (NT>0 and vabs(val) > vabs(Hbuffer[0])):
#insertion de val dans HBuffer
            if(vabs(val) >= vabs(Hbuffer[end])):
                insert(Hbuffer,end,val)
                int_insert(Hindex,end,i)
            else:
#recherche de l'index a inserer par dichotomie
                index = (NT+1)//2
                offset = index
                while(True):
                    if(index >= NT):
                        index = NT-1
                    if(index <= 0):
                        index = 1
                    if(vabs(Hbuffer[index-1]) <= vabs(val) and vabs(val) <= vabs(Hbuffer[index])):
                        insert(Hbuffer,index-1,val)
                        int_insert(Hindex,index-1,i)
                        break
                    offset = (offset+1)//2
                    if(vabs(val) > vabs(Hbuffer[index])):
                        index += offset
                    else:
                        index -= offset
        #dot += val

    #on soustrait a dot les NT valeurs les plus grosses
    #for i in range(NT):
    #    dot -= Hbuffer[i]
    qsort(Hindex,cpNT,sizeof(int),compare_ints)
    cdef int count = 0
    for i in range(N):
        if(count >= cpNT or i!=Hindex[count]):
            dot += X1[i*incx]*X2[i*incy]
        else:
            count += 1
    
    return dot

    

def tdot(X1,X2,NT):
#dans le cas ou X2 et à une dimension on le passe en 2D (colonne)
#dans le cas ou X1 et à une dimension on le passe en 2D (line)
#on fait le produit
#on renvoie un array
    TX2=X2
    TX1=X1
    if(len(X2.shape)==1):
        TX2=np.zeros((len(X2),1))
        for i in range(len(X2)):
            TX2[i,0] = X2[i]
    if(len(X1.shape)==1):
        TX1=np.zeros((1,(len(X1))))
        for i in range(len(X1)):
            TX1[0,i] = X1[i]

    TX2 = np.asfortranarray(TX2)
    TX1 = np.ascontiguousarray(TX1)

    Res = tdotaux(TX1,TX2,NT)

    if(len(X2.shape)==1 and len(X1.shape)==1):
        Res = Res[0,0]
    else:
        if(len(X2.shape)==1):
            TRes = np.zeros((TX1.shape[0]))
            for i in range(TX1.shape[0]):
                TRes[i] = Res[i,0]
            Res=TRes
        if(len(X1.shape)==1):
            TRes = np.zeros((TX2.shape[1]))
            for i in range(TX2.shape[1]):
                TRes[i] = Res[0,i]
            Res=TRes

    return Res

def tdotaux(cnp.ndarray[double,ndim=2,mode='c'] X1, cnp.ndarray[double,ndim=2,mode='fortran'] X2, int NT):

    s1 = X1.shape
    s2 = X2.shape

    cdef int lin1
    cdef int lin2
    cdef int col1
    cdef int col2
    
    lin1 = s1[0]
    col1 = s1[1]
    lin2 = s2[0]
    col2 = s2[1]
    Res = np.zeros((lin1,col2))
    
    if(col1!=lin2):
        raise ValueError("wrond dimensions")
    
    cdef int i
    cdef int j
    for i in range(lin1):
        for j in range(col2):
            Res[i][j] = cvtdot(<double *>(X1.data + col1*i*sizeof(double)),<double *>(X2.data + col1*j*sizeof(double)),col1,NT)
    return Res

def project(M,eps):
    W, V = linalg.eigh(M)
#le: valeur par laquelle on remplace les vp negatives de M
    eps = 0.1
    _p_ = False
    for i in range(len(W)):
        if(W[i] < 0):
            W[i] = eps
            _p_ = True
    M = np.dot(np.dot(V,np.diag(W)),V.T)
    if _p_:
        print("singular matrix projected")
    return M





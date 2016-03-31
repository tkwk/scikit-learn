# -*- coding: utf-8 -*-

cimport numpy as cnp
from libc.math cimport isfinite
import numpy as np
from numpy import linalg
import math
from libc.stdlib cimport malloc, free

cdef double * Hbuffer = NULL
cdef int Hsize = 0

cdef inline double vabs(double x):
    if(x>=0):
        return x
    return -x

cdef void insert(double * H, int index, double value):
    cdef int i
    for i in range(index):
        H[i] = H[i+1]
    H[index] = value

cdef double cvtdot(double * X1, double * X2, int N, int NT):
    global Hbuffer, Hsize
#on ajuste la taille du buffer si necessaire
    if(NT > Hsize):
        if(Hbuffer != NULL):
            free(Hbuffer)
        Hbuffer = <double *> malloc(sizeof(double)*NT)

#calcul du produit scalaire
    cdef double dot = 0.0
    cdef int i
    cdef double val = 0.0
    cdef int index
    cdef int offset
    cdef int end = NT-1

    for i in range(NT):
        Hbuffer[i] = 0.0

    for i in range(N):
        if( (not isfinite(X1[i])) or (not isfinite(X2[i])) ):
            insert(Hbuffer,end,0.0)
            end -= 1
            NT -= 1
            continue
        val = X1[i]*X2[i]
        if (NT>0 and vabs(val) > vabs(Hbuffer[0])):
#insertion de val dans HBuffer
            if(vabs(val) >= vabs(Hbuffer[end])):
                insert(Hbuffer,end,val)
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
                        break
                    offset = (offset+1)//2
                    if(vabs(val) > vabs(Hbuffer[index])):
                        index += offset
                    else:
                        index -= offset
        dot += val

    #on soustrait a dot les NT valeurs les plus grosses
    for i in range(NT):
        dot -= Hbuffer[i]
    return dot

def tdot(X1,X2,NT):
#dans le cas ou X2 et à une dimension on le passe en 2D (colonne)
#on fait le produit
#on renvoie un array
    TX2=X2
    if(len(X2.shape)==1):
        TX2=np.zeros((len(X2),1))
        for i in range(len(X2)):
            TX2[i,0] = X2[i]

    TX2 = np.asfortranarray(TX2)
    X1 = np.ascontiguousarray(X1)

    Res = tdotaux(X1,TX2,NT)

    if(len(X2.shape)==1):
        TRes = np.zeros((X1.shape[0]))
        for i in range(X1.shape[0]):
            TRes[i] = Res[i,0]
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
    for i in range(len(W)):
        if(W[i] < 0):
            W[i] = eps
    M = np.dot(np.dot(V,np.diag(W)),V.T)
    return M





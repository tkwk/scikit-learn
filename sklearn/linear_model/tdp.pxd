# -*- coding: utf-8 -*-

cimport numpy as cnp
from libc.math cimport isfinite
import numpy as np
from numpy import linalg
import math
from libc.stdlib cimport malloc, free

cdef double * Hbuffer
cdef int * Hind
cdef int Hsize
cdef int gNT

cdef inline double vabs(double x) nogil

cdef void insert(double * H, int index, double value) nogil

cdef double cvtdot(double * X1, double * X2, int N, int NT) nogil

cdef double tddot(int N, double * X1, int incx, double * X2, int incy) nogil

cdef double tddot_index(int N, double * X1, int incx, double * X2, int incy, int * index) nogil

cdef double ptddot(int N, double * X1, int incx, double * X2, int incy, int NT, int * iarray = *) nogil

cdef void trimm(int N, double * X, int incx, int NT) nogil

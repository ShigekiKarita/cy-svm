# distutils: language = c++
# distutils: sources = svm.cpp

import numpy as np
cimport numpy as np
from libcpp cimport bool
ctypedef np.float64_t DOUBLE_t


cdef extern from "svm.h":
    cdef cppclass SVM:
        void fit(double* train_set, double* target_set, size_t n, size_t d,
                 double c, double eps, size_t loop_limit, bool is_linear)
        double decision_function(double* input)


cdef class SVMWrapper:
    cdef SVM * _thisptr

    def __cinit__(self):
        self._thisptr = new SVM()
        if self._thisptr == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    def fit(self, np.ndarray[DOUBLE_t, ndim=2] train_set, np.ndarray[DOUBLE_t, ndim=1] target_set,
                   double c=1.0, double eps=1e-3, size_t loop_limit=1000, bool is_linear=False):
        cdef int n = len(train_set)
        cdef int d = len(train_set[0])
        self._thisptr.fit(<double *>train_set.data, <double *>target_set.data,
                          n, d, c, eps, loop_limit, is_linear)
        return self

    def df_one(self, np.ndarray[DOUBLE_t, ndim = 1] test_set):
        return self._thisptr.decision_function(<double *>test_set.data)

    def df_batch(self, test_set):
        return np.array([self.df_one(t) for t in test_set])

    def decision_function(self, test_set):
        cdef int ndim = np.ndim(test_set)
        if ndim == 1:
            return self.df_one(test_set)
        elif ndim == 2:
            return self.df_batch(test_set)
        else:
            raise NotImplementedError("[Error] len(test_set.sshape): %d should be 1 or 2. How about reshape it?" % ndim)
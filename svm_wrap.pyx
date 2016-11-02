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
        double predict(double* input)

cdef class SVMWrapper:
    cdef SVM * _thisptr

    def __cinit__(self):
        self._thisptr = new SVM()
        if self._thisptr == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    def _fit_impl(self, np.ndarray[DOUBLE_t, ndim=2] train_set, np.ndarray[DOUBLE_t, ndim=1] target_set,
                  size_t n, size_t d, double c, double eps, size_t limit, bool is_linear):
        self._thisptr.fit(<double *>train_set.data, <double *>target_set.data, n, d, c, eps, limit, is_linear)
        return self

    def fit(self, xs, ts, c=1e3, eps=1e-3, loop_limit=1000, is_linear=False):
        return self._fit_impl(xs, ts, len(xs), len(xs[0]), c, eps, loop_limit, is_linear)

    def predict_one(self, np.ndarray[DOUBLE_t, ndim = 1] test_set):
        return  self._thisptr.predict(<double *>test_set.data)

    def predict_batch(self, test_set):
        return np.array([self.predict_one(t) for t in test_set])

    def predict(self, test_set):
        rank = len(test_set.shape)
        if rank == 1:
            return self.predict_one(test_set)
        elif rank == 2:
            return self.predict_batch(test_set)
        else:
            raise NotImplementedError("[Error] len(test_set.sshape): %d should be 1 or 2. How about reshape it?" % rank)
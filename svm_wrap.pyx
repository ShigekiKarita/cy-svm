# distutils: language = c++
# distutils: sources = svm.cpp

import numpy as np
cimport numpy as np
cimport cython

DOUBLE = np.float64
ctypedef np.float64_t DOUBLE_t


# import external C implementation
cdef extern from "svm.h":
    cdef cppclass SVM:
        void fit(double* train_set, double* target_set, size_t n, size_t d, size_t loop_limit)
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

    def _fit_impl(self, np.ndarray[DOUBLE_t, ndim=2] train_set,
                  np.ndarray[DOUBLE_t, ndim=1] target_set,
                  size_t n, size_t d, size_t limit):
        #n, d = train_set.shape
        self._thisptr.fit(<double *>train_set.data, <double *>target_set.data, n, d, limit)
        return self

    def fit(self, xs, ts, loop_limit=1000000):
        return self._fit_impl(xs, ts, len(xs), len(xs[0]), loop_limit)

    def predict_one(self, np.ndarray[DOUBLE_t, ndim = 1] test_set):
        return  self._thisptr.predict(<double *>test_set.data)

    def predict_batch(self, test_set):
        n = len(test_set)
        res = np.empty(n)
        for i in range(n):
            res[i] = self.predict_one(test_set[i])
        return res

    def predict(self, test_set):
        rank = len(test_set.shape)
        if rank == 1:
            return self.predict_one(test_set)
        elif rank == 2:
            return self.predict_batch(test_set)
        else:
            raise NotImplementedError("len(test_set.sshape): %d should be 1 or 2" % rank)
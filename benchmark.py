from timeit import timeit
from test import DecisionBoundary
import svm_wrap

from sklearn.svm import SVC, LinearSVC


class Benchmark:
    def __init__(self, **kwargs):
        self.d = DecisionBoundary(**kwargs)
        self.c = 1.0
        self.gamma = 2.0
        self.max_iter = 1e3
        self.tol = 1e-3

    def fit_cy(self):
        self.cy = svm_wrap.SVMWrapper()
        self.d.fit(self.cy, c=self.c, is_linear=False, loop_limit=self.max_iter, eps=self.tol)
        return self

    def fit_sk(self):
        self.sk = SVC(C=self.c, gamma=self.gamma, max_iter=self.max_iter, tol=self.tol, shrinking=False)
        self.d.fit(self.sk)
        return self

    def accuracy(self):
        print("=== cy-svm result ===")
        self.d.accuracy(self.cy)
        print("=== sklearn result ===")
        self.d.accuracy(self.sk)


def measure(stmt):
    setup = "from benchmark import Benchmark"
    number = 100
    t = timeit(stmt=stmt, setup=setup, number=number)
    print("timtit: {}\n> {} sec {} trials".format(stmt, t, number))


if __name__ == '__main__':
    n = 100
    d = 2
    bench = "Benchmark(num=%d, dim=%d)." % (n, d)
    measure(bench + "fit_sk()")
    measure(bench + "fit_cy()")

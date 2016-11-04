from test import DecisionBoundary
import svm_wrap

from sklearn.svm import SVC, LinearSVC

class Benchmark:
    def __init__(self):
        self.d = DecisionBoundary()
        self.c = 1.0
        self.gamma = 2.0
        self.max_iter = 1e3
        self.tol = 1e-3

    def fit_cy(self):
        self.cy = svm_wrap.SVMWrapper()
        self.d.fit(self.cy, c=self.c, is_linear=False, loop_limit=self.max_iter, eps=self.tol)

    def fit_sk(self):
        self.sk = SVC(C=self.c, gamma=self.gamma, max_iter=self.max_iter, tol=self.tol)
        self.d.fit(self.sk)

    def accuracy(self):
        print("=== cy-svm result ===")
        self.d.accuracy(self.cy)
        print("=== sklearn result ===")
        self.d.accuracy(self.sk)


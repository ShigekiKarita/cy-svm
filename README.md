# Support Vector Machines for Cython

a basic example using C++ class in Cython

usage:

```
$ python setup.py install
$ python setup.py test # plot classification results
$ python setup.py build
timtit: Benchmark().fit_cy()
> 0.9718976389849558 sec 1000 trials
timtit: Benchmark().fit_sk()
> 0.44173674198100343 sec 1000 trials
```

![svms](res/svms.png)
![result](res/result.png)

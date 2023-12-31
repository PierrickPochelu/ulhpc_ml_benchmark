# ulhpc_ml_benchmark
Benchmark of many ML algorithms at once for easy platform evaluation.

This repo aims evaluating +40 Machine Learning algorithms in training and inference modes. It makes easier the evaluation of platforms based on the performance of realistic algorithms instead of using FLOPS or other CPU characteristics. Algorithms have different complexity (in Big-O notation) and may require several order of magnitude of time. This is why we structure the benchmark to assess the number of data samples processed within a fixed amount of time, rather than measuring the computational time for a fixed quantity of data. The latter approach would be impractical due to the vast differences in processing times—ranging from milliseconds to hours—across various algorithms.

Utilization example:
```python
(base) pierrick@LinuxUniBXD7LS3:~/project/ulhpc_ml_benchmark$ python3
Python 3.10.9 (main, Jan 11 2023, 15:21:40) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from bench import bench
>>> bench(num_samples=100, num_features=10, fix_comp_time=1)
```
bench(num_samples=100, num_features=10, fix_comp_time=1) represents a benchmark on 100 data points, 10 features per point, and each algorithm is trained 1 second. 

After ~40 seconds the output is:

```
>>> bench(100,10,1)
TheilSenRegressor 10 34716
ElasticNetCV 18 34552
LassoCV 18 34426
RandomForestRegressor 25 641
ExtraTreesRegressor 39 462
GradientBoostingRegressor 59 8848
MLPRegressor 86 19242
HistGradientBoostingRegressor 101 1297
QuantileRegressor 142 34948
BaggingRegressor 148 1665
HuberRegressor 173 33674
RANSACRegressor 918 23176
AdaBoostRegressor 1393 8308
GaussianProcessRegressor 1445 6168
LassoLarsIC 1859 34741
ARDRegression 1938 33936
TransformedTargetRegressor 2003 17661
Lasso 2041 31807
ElasticNet 2071 31970
RidgeCV 2391 35068
TweedieRegressor 2409 34881
GammaRegressor 2476 33955
PoissonRegressor 2549 29877
BayesianRidge 2600 33377
PLSRegression 3407 28063
NuSVR 3473 21918
Ridge 3580 34927
LassoLars 3619 34460
SGDRegressor 3718 33920
Lars 3723 34997
OrthogonalMatchingPursuit 3887 35504
KernelRidge 3921 6713
SVR 4079 21216
LinearRegression 4674 35220
PassiveAggressiveRegressor 4792 34678
LinearSVR 5054 35520
RadiusNeighborsRegressor 5446 992
KNeighborsRegressor 5537 4052
ExtraTreeRegressor 6016 22086
DecisionTreeRegressor 6310 22741
DummyRegressor 13205 200102
```

For each algorithm line, the benchmark output provides the following information:
* Algorithm name
* Number of time the algorithm ingest batch of `num_samples` during training `fix_comp_time`
* Same for inference

Notice: the line are sorted according the 1st column.

Example of experiments on 4 CPUs:

![Result](result.png)

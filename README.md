# ulhpc_ml_benchmark
Benchmark of many ML algorithms at once for easy platform evaluation.

This repo aims evaluating +40 Machine Learning algorithms in training and inference modes. It makes easier the evaluation of platforms based on the performance of realistic algorithms instead of using FLOPS or other CPU characteristics. Algorithms have different complexity (in Big-O notation) and may require several order of magnitude of time. This is why we structure the benchmark to assess the number of data samples processed within a fixed amount of time, rather than measuring the computational time for a fixed quantity of data. The latter approach would be impractical due to the vast differences in processing times—ranging from milliseconds to hours—across various algorithms.

code:
```python
bench(num_samples=100, num_features=100, fix_comp_time=1)
```
For each algorithm, the benchmark output provides the following information:

*    Algorithm name
*    Rate of data ingestion during training (samples per fixed computation time)
* Rate of data ingestion during inference

output:
```
LassoCV 6 39640
ElasticNetCV 7 40089
RandomForestRegressor 26 657
ExtraTreesRegressor 47 759
GradientBoostingRegressor 74 11731
MLPRegressor 75 20647
HistGradientBoostingRegressor 125 1634
BaggingRegressor 185 1975
HuberRegressor 257 39302
QuantileRegressor 750 38637
ARDRegression 1553 39902
Lasso 1603 36295
ElasticNet 1650 36593
AdaBoostRegressor 1733 10175
BayesianRidge 2058 37766
PoissonRegressor 2234 36531
TransformedTargetRegressor 2306 21393
TweedieRegressor 2438 37803
GaussianProcessRegressor 2464 20872
GammaRegressor 2580 39750
RidgeCV 2689 37955
TheilSenRegressor 3588 40284
PLSRegression 3893 31798
Ridge 3945 38807
SGDRegressor 4045 37852
LassoLars 4134 39915
OrthogonalMatchingPursuit 4188 38354
Lars 4249 38947
NuSVR 4795 24477
SVR 5165 24685
LinearRegression 5479 41247
PassiveAggressiveRegressor 5488 38287
KernelRidge 5787 9623
LinearSVR 6077 41341
ExtraTreeRegressor 7135 27034
DecisionTreeRegressor 7145 26169
RadiusNeighborsRegressor 9689 2168
KNeighborsRegressor 9707 9249
DummyRegressor 15243 234076
```

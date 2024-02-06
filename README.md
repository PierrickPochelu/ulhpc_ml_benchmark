# ulhpc_ml_benchmark
Benchmark of many ML algorithms at once for easy platform evaluation.

This repo aims evaluating +40 Machine Learning algorithms in training and inference modes. It makes easier the evaluation of platforms based on the performance of realistic algorithms instead of using FLOPS or other CPU characteristics. Algorithms have different complexity (in Big-O notation) and may require several order of magnitude of time. This is why we structure the benchmark to assess the number of data samples processed within a fixed amount of time, rather than measuring the computational time for a fixed quantity of data. The latter approach would be impractical due to the vast differences in processing times—ranging from milliseconds to hours—across various algorithms.

Utilization example:
```python
(base) pierrick@LinuxUniBXD7LS3:~/project/ulhpc_ml_benchmark$ python3
Python 3.10.9 (main, Jan 11 2023, 15:21:40) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from bench import bench
>>> bench(num_samples=1000, num_features=100, fix_comp_time=1, reg_or_cls="reg")
```
bench(num_samples=100, num_features=10, fix_comp_time=1) represents a benchmark on 100 data points, 10 features per point, and each algorithm is trained 1 second. 

After ~50 seconds the output is:

```
LassoCV 0.234817 16749
ElasticNetCV 0.33169 11628
TheilSenRegressor 1.527735 18588
MultiTaskElasticNetCV 6.302748 17154
MultiTaskLassoCV 6.324106 17001
QuantileRegressor 6.912841 18286
GaussianProcessRegressor 19 24
RandomForestRegressor 25 535
MLPRegressor 26 4021
ExtraTreesRegressor 39 598
GradientBoostingRegressor 56 4370
MultiTaskLasso 60 15964
MultiTaskElasticNet 61 17405
HistGradientBoostingRegressor 65 1073
HuberRegressor 80 17127
BaggingRegressor 148 723
NuSVR 148 14297
ElasticNet 170 11316
KernelRidge 181 866
Lasso 219 16441
RidgeCV 377 20126
ARDRegression 416 18103
BayesianRidge 436 17228
RANSACRegressor 565 14170
SGDRegressor 768 11772
LassoLarsIC 782 17114
AdaBoostRegressor 1033 5163
PassiveAggressiveRegressor 1277 18707
TweedieRegressor 1304 11981
TransformedTargetRegressor 1312 7764
PLSRegression 1522 4397
OrthogonalMatchingPursuit 1549 17003
LassoLars 1592 16402
Ridge 1621 19856
Lars 1673 17026
GammaRegressor 1812 16029
PoissonRegressor 1915 17134
LinearSVR 1916 18146
SVR 2147 13748
LinearRegression 2219 17062
ExtraTreeRegressor 4364 11644
DecisionTreeRegressor 4687 12640
KNeighborsRegressor 7255 280
RadiusNeighborsRegressor 7347 63
DummyRegressor 13247 185039
```

Now let's benchmark classifiers

```python
>>> bench(num_samples=1000, num_features=100, fix_comp_time=1, reg_or_cls="cls")
```

returns

```
GaussianProcessClassifier 12 29
RandomForestClassifier 17 323
ExtraTreesClassifier 23 338
AdaBoostClassifier 24 158
GradientBoostingClassifier 32 4082
MLPClassifier 38 3886
SVC 40 24
LogisticRegressionCV 44 15586
HistGradientBoostingClassifier 71 1216
LabelSpreading 109 164
BaggingClassifier 123 715
CalibratedClassifierCV 124 642
LabelPropagation 141 165
CategoricalNB 208 976
RidgeClassifierCV 285 14668
QuadraticDiscriminantAnalysis 437 1805
SGDClassifier 448 15471
Perceptron 701 16004
RidgeClassifier 742 14513
LinearSVC 804 16303
BernoulliNB 858 2100
ComplementNB 1073 10361
PassiveAggressiveClassifier 1155 15967
LogisticRegression 1167 15897
MultinomialNB 1655 13369
GaussianNB 1887 3059
DecisionTreeClassifier 2519 11871
ExtraTreeClassifier 2751 11878
NearestCentroid 3846 3399
RadiusNeighborsClassifier 5069 96
KNeighborsClassifier 5436 548
DummyClassifier 15287 121029
```


For each algorithm line, the benchmark output provides the following information:
* Algorithm name
* Data points ingested per unit of time during training. Units are `num_samples` per `fix_comp_time` seconds.
* Data points ingested per unit of time during inference.

Notice: the line are sorted according the 1st column.




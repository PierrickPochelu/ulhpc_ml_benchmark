> ⚠️ **Warning: This repository is obsolete.**  
> Development has moved to a new location. Please visit the updated repository here: [https://github.com/PierrickPochelu/scikit-learn-bench](https://github.com/PierrickPochelu/scikit-learn-bench)

# ulhpc_ml_benchmark
Benchmark of many ML algorithms at once for easy platform evaluation.

This repo aims evaluating +40 Machine Learning algorithms in training and inference modes. It makes easier the evaluation of platforms based on the performance of realistic algorithms instead of using FLOPS or other CPU characteristics. Algorithms have different complexity (in Big-O notation) and may require several order of magnitude of time. This is why we structure the benchmark to assess the number of data samples processed within a fixed amount of time, rather than measuring the computational time for a fixed quantity of data. The latter approach would be impractical due to the vast differences in processing times—ranging from milliseconds to hours—across various algorithms.
## Installation
```python
pip install ulhpc_ml_benchmark
```
The only dependency is scikit-learn

## API
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
ElasticNetCV                  0.202  17985  
LassoCV                       0.211  17338  
TheilSenRegressor             1.651  16240  
MultiTaskLassoCV              5.284  17517  
MultiTaskElasticNetCV         6.001  16638  
QuantileRegressor             10.691 11873  
MLPRegressor                  14     2957   
GaussianProcessRegressor      21     26     
RandomForestRegressor         23     541    
ExtraTreesRegressor           42     614    
GradientBoostingRegressor     61     4987   
HistGradientBoostingRegressor 64     1119   
MultiTaskElasticNet           67     16727  
MultiTaskLasso                70     15296  
HuberRegressor                85     18049  
NuSVR                         146    13873  
BaggingRegressor              153    743    
KernelRidge                   174    869    
ElasticNet                    205    17569  
Lasso                         219    15988  
RidgeCV                       265    11561  
ARDRegression                 332    11616  
BayesianRidge                 370    15134  
RANSACRegressor               424    9158   
SGDRegressor                  684    11609  
LassoLarsIC                   843    16636  
Ridge                         1012   11731  
AdaBoostRegressor             1074   5504   
PassiveAggressiveRegressor    1255   11666  
TransformedTargetRegressor    1299   10994  
PoissonRegressor              1372   10837  
Lars                          1631   18324  
LassoLars                     1652   17542  
OrthogonalMatchingPursuit     1657   16939  
PLSRegression                 1725   4588   
TweedieRegressor              1856   16051  
LinearSVR                     1911   11702  
GammaRegressor                2041   17493  
SVR                           2165   13647  
LinearRegression              2175   16603  
ExtraTreeRegressor            4934   12517  
DecisionTreeRegressor         5088   12511  
RadiusNeighborsRegressor      6877   63     
KNeighborsRegressor           7749   288    
DummyRegressor                13801  192216
```

Now let's benchmark classifiers

```python
>>> bench(num_samples=1000, num_features=100, fix_comp_time=1, reg_or_cls="cls")
```

displays:

```
GaussianProcessClassifier      0.103 24     
RandomForestClassifier         16    296    
AdaBoostClassifier             20    129    
ExtraTreesClassifier           22    301    
GradientBoostingClassifier     26    3466   
MLPClassifier                  33    3466   
SVC                            35    22     
LogisticRegressionCV           40    11889  
HistGradientBoostingClassifier 48    912    
LabelSpreading                 96    148    
BaggingClassifier              108   536    
CalibratedClassifierCV         112   617    
LabelPropagation               127   147    
CategoricalNB                  177   867    
RidgeClassifierCV              270   12418  
Perceptron                     397   13088  
QuadraticDiscriminantAnalysis  422   1548   
SGDClassifier                  440   12716  
RidgeClassifier                688   12465  
LinearSVC                      759   12484  
LogisticRegression             1031  12911  
PassiveAggressiveClassifier    1072  12447  
BernoulliNB                    1137  2816   
ComplementNB                   1280  11883  
MultinomialNB                  1498  11434  
GaussianNB                     1649  2609   
DecisionTreeClassifier         2357  10175  
ExtraTreeClassifier            2407  10034  
NearestCentroid                3245  2938   
RadiusNeighborsClassifier      4481  93     
KNeighborsClassifier           4584  296    
DummyClassifier                13901 107307 
```


For each algorithm line, the benchmark output provides the following information:
* Algorithm name
* Data points ingested per unit of time during training. Units are `num_samples` per `fix_comp_time` seconds.
* Data points ingested per unit of time during inference.

Notice: the line are sorted according the 1st column.




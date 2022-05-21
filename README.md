# Sorted Shift: A large-scale benchmark for out-of-distribution model predictive performance evaluation under gradual distribution shift

**Abstract**
Out-of-distribution evaluation of machine learning models has relied on individual datasets, making experimentation results fragile and preventing researchers from testing new algorithmic proposals. Our work aims to adapt an existing large-scale data benchmark for machine learning to have out-of-distribution data, helping to provide statistical evidence for new algorithms without the burden of cleaning, pre-processing, and validating a large number of datasets. In order to evaluate model performance on out-of-distribution data, we iteratively sort each column of the data, split the data into three equal parts, train in the middle one, and evaluate the other two. This data partitioning allows decomposing the model performance on three different errors: an already seen data (train error), a hold-out set with unseen data from the same distribution (test error), and statistically new data (out-of-distribution error). 

### Vanilla code snippet

```python
from benchmark_experiment import OODbenchmark
from sklearn.linear_model import LogisticRegression

OODbenchmark(
        datasets="adult",
        model=LogisticRegression(),
        classification=True
    )
```
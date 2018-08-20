---
layout: post
title:  'feature_stuff: ML lib for advanced feature processing'
---

## feature_stuff: a python machine learning library for advanced feature extraction, processing and interpretation.

| Latest Release | [see on pypi.org](https://pypi.org/project/feature-stuff/) |
| Package Status | [see on pypi.org](https://pypi.org/project/feature-stuff/) |
| License | [see on github](https://github.com/hiflyin/Feature-Stuff/blob/master/LICENSE) |
| Build Status | [see on pypi.org](https://travis-ci.org/hiflyin/Feature-Stuff/) |


## What is it

**feature_stuff** is a Python package providing fast and flexible algorithms and functions
for extracting, processing and interpreting features.

For installation instructions and API see the links above.

## What you can do with it

For example you can extract and interpret interactions between features in large data-sets with the following function:

### feature_stuff.add_interactions

    Inputs:
        df: a pandas dataframe
        model: boosted trees model (currently xgboost supported only). Can be None in which case the interactions have
        to be provided
        interactions: list in which each element is a list of features/columns in df, default: None

    Output: df containing the group values added to it


Example on extracting interactions from tree based models and adding them as new features to your dataset.

```python
import feature_stuff as fs
import pandas as pd
import xgboost as xgb

data = pd.DataFrame({"x0":[0,1,0,1], "x1":range(4), "x2":[1,0,1,0]})
print data
   x0  x1  x2
0   0   0   1
1   1   1   0
2   0   2   1
3   1   3   0

target = data.x0 * data.x1 + data.x2*data.x1
print target.tolist()
[0, 1, 2, 3]

model = xgb.train({'max_depth': 4, "seed": 123}, xgb.DMatrix(data, label=target), num_boost_round=2)
fs.addInteractions(data, model)

## at least one of the interactions in target must have been discovered by xgboost
print data
   x0  x1  x2  inter_0
0   0   0   1        0
1   1   1   0        1
2   0   2   1        0
3   1   3   0        3

## if we want to inspect the interactions extracted
from feature_stuff import model_features_insights_extractions as insights
print insights.get_xgboost_interactions(model)
[['x0', 'x1']]

```

Another example is extracting complex target encodings from categorical features and adding them as new features to your dataset.

### feature_stuff.target_encoding

    Inputs:
        df: a pandas dataframe containing the column for which to calculate target encoding (categ_col)
        ref_df: a pandas dataframe containing the column for which to calculate target encoding and the target (y_col)
            for example we might want to use train data as ref_df to encode test data
        categ_col: the name of the categorical column for which to calculate target encoding
        y_col: the name of the target column, or target variable to predict
        smoothing_func: the name of the function to be used for calculating the weights of the corresponding target
            value inside ref_df. Default: exponentialPriorSmoothing.
        aggr_func: the statistic used to aggregate the target variable values inside each category of the categ_col
        smoothing_prior_weight: a prior weight to put on each category. Default 1.

    Output: df containing a new column called <categ_col + "_bayes_" + aggr_func> containing the encodings of categ_col



```python
import feature_stuff as fs
import pandas as pd

train_data = pd.DataFrame({"x0":[0,1,0,1]})
test_data = pd.DataFrame({"x0":[1, 0, 0, 1]})
target = range(4)

train_data = fs.target_encoding(train_data, train_data, "x0", target, smoothing_func=fs.exponentialPriorSmoothing,
                                        aggr_func="mean", smoothing_prior_weight=1)
test_data = fs.target_encoding(test_data, train_data, "x0", target, smoothing_func=fs.exponentialPriorSmoothing,
                                        aggr_func="mean", smoothing_prior_weight=1)

##train data with target encoding of "x0"
print(train_data)
   x0  y_xx  g_xx  x0_bayes_mean
0   0     0     0       1.134471
1   1     1     0       1.865529
2   0     2     0       1.134471
3   1     3     0       1.865529

##test data with target encoding of "x0"
print(test_data)
   x0  x0_bayes_mean
0   1       1.865529
1   0       1.134471
2   0       1.134471
3   1       1.865529

```

### feature_stuff.cv_target_encoding

    Inputs:
        df: a pandas dataframe containing the column for which to calculate target encoding (categ_col) and the target
        categ_cols: a list or array with the the names of the categorical columns for which to calculate target encoding
        y_col: a numpy array of the target variable to predict
        cv_folds: a list with fold pairs as tuples of numpy arrays for cross-val target encoding
        smoothing_func: the name of the function to be used for calculating the weights of the corresponding target
            value inside ref_df. Default: exponentialPriorSmoothing.
        aggr_func: the statistic used to aggregate the target variable values inside each category of the categ_col
        smoothing_prior_weight: a prior weight to put on each category. Default 1.
        verbosity: 0-none, 1-high_level, 2-detailed

    Output: df containing a new column called <categ_col + "_bayes_" + aggr_func> containing the encodings of categ_col

See feature_stuff.target_encoding example above.


## Contributing to feature-stuff

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome.




        
___
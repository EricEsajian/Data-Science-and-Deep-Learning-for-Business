#!/usr/bin/env python
# coding: utf-8

# <p style="padding: 10px; border: 1px solid black;">
# <img src=".././images/MLU-NEW-logo.png" alt="drawing" width="400"/> <br/>
# 
# # MLU Day One Machine Learning - Walkthrough & Advanced AutoGluon Features

# # Part I - Walkthrough & Discussions
# Now that you have finished your hands-on activity, let's walk through the code you have used and discuss it. <br/>

# In[1]:


# Importing the newly installed AutoGluon code library
from autogluon.tabular import TabularPredictor, TabularDataset

# READING IN THE DATA
train = TabularDataset(".././datasets/training.csv")
mlu_test_data = TabularDataset(".././datasets/mlu-leaderboard-test.csv")

# TRAINING THE MODEL
predictor = TabularPredictor(label="Price").fit(train_data=train, time_limit=60)

# GENERATRING PREDICTIONS
predictions = predictor.predict(mlu_test_data)

# WRITE PREDICTIONS TO A NEW FILE
# Creating a new dataframe for the submission
submission = mlu_test_data[["ID"]].copy(deep=True)

# Creating label column from price prediction list
submission["Price"] = predictions

# Saving our csv file for Leaderboard submission
# index=False prevents printing the row IDs as separate values
submission.to_csv(
    ".././datasets/predictions/Solution-Demo.csv",
    index=False,
)


# ---
# 
# # Part II - Advanced AutoGluon Features
# 
# ## ML Problem Description
# Predict the occupation of individuals using census data. 
# > This is a multiclass classification task (15 distinct classes). <br>
# 
# For the advanced feature demonstration we want to use a new dataset: Census data. In this particular dataset, each row corresponds to an individual person, and the columns contain various demographic characteristics collected for the census.
# 
# We’ll predict the occupation of an individual - this is a multiclass classification problem. Start by importing AutoGluon’s `TabularPredictor` and `TabularDataset`, and load the data from a S3 bucket.

# In[2]:


get_ipython().system('pip install -q bokeh==2.0.1')


# ### Loading the data

# In[3]:


from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import numpy as np

# Load in the dataset
train_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
# Subsample a subset of data for faster demo, try setting this to much larger values
subsample_size = 5000
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()


# ### Setting the target
# 

# In[4]:


# Assign column that contains the label to a variable that can be re-used later
label = "occupation"

print("Summary of occupation column: \n")
train_data["occupation"].describe()


# ### Train, validation, test split

# In[5]:


# Create a train & validation split
train_data, val_data = train_test_split(
    train_data, test_size=0.1, shuffle=True, random_state=23
)

# Let's load the test data
test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")

# We need to split the test dataset into a features and a label subset
y_test = test_data[label]
test_data_nolabel = test_data.drop(columns=[label])  # delete label column


# ### Specifying performance metric

# In[6]:


# We specify eval-metric just for demo (unnecessary as it's the default)
metric = "accuracy"


# The full list of parameters can be found here:
# 
# `'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted', 'roc_auc', 'average_precision', 'precision', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score'`

# ### Specifying hyperparameters and tuning them

# In[7]:


import autogluon.core as ag

# Set Neural Net options
# Specifies non-default hyperparameter values for neural network models
nn_options = {
    # number of training epochs (controls training time of NN models)
    "num_epochs": 10,
    # learning rate used in training (real-valued hyperparameter searched on log-scale)
    "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
    # activation function used in NN (categorical hyperparameter, default = first entry)
    "activation": ag.space.Categorical("relu", "softrelu", "tanh"),
    # each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer to use
    "layers": ag.space.Categorical([100], [1000], [200, 100], [300, 200, 100]),
    # dropout probability (real-valued hyperparameter)
    "dropout_prob": ag.space.Real(0.0, 0.5, default=0.1),
}

# Set GBM options
# Specifies non-default hyperparameter values for lightGBM gradient boosted trees
gbm_options = {
    # number of boosting rounds (controls training time of GBM models)
    "num_boost_round": 100,
    # number of leaves in trees (integer hyperparameter)
    "num_leaves": ag.space.Int(lower=26, upper=66, default=36),
    "depth": [2,4,8,10]
}

# Add both NN and GBM options into a hyperparameter dictionary
# hyperparameters of each model type
# When these keys are missing from the hyperparameters dict, no models of that type are trained
hyperparameters = {
    "GBM": gbm_options,
    "NN": nn_options,
}

# Train various models for ~2 min
time_limit = 2 * 60
# Number of trials for hyperparameters
num_trials = 5

# To tune hyperparameters using Bayesian optimization to find best combination of params
search_strategy = "auto"

# HPO is not performed unless hyperparameter_tune_kwargs is specified
hyperparameter_tune_kwargs = {
    "num_trials": num_trials,
    "scheduler": "local",
    "searcher": search_strategy,
}


# ### Specifying settings for TabularPredictor

# In[8]:


# Train various models for ~2 min
time_limit = 2 * 60
# Number of trials for hyperparameters
num_trials = 5


# ### Train Model using TabularPredictor

# In[9]:


predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    tuning_data=val_data,
    time_limit=time_limit,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)


# ### Predict on the test data

# In[10]:


y_pred = predictor.predict(test_data_nolabel)
print(f"Predictions:  {list(y_pred)[:5]}")
perf = predictor.evaluate(test_data, auxiliary_metrics=False)


# Use the following to view a summary of what happened during the fit. Now this command will show details of the hyperparameter-tuning process for each type of model:

# In[11]:


predictor.fit_summary()


# In the above example, the predictive performance may be poor because we are using few training datapoints and small ranges for hyperparameters to ensure quick runtimes. You can call `fit()` multiple times while modifying these settings to better understand how these choices affect performance outcomes. For example: you can increase `subsample_size` to train using a larger dataset, increase the `num_epochs` and `num_boost_round` hyperparameters, and increase the `time_limit` (which you should do for all code in these tutorials). To see more detailed output during the execution of `fit()`, you can also pass in the argument: `verbosity = 3`.

# ### Model ensembling with stacking/bagging
# Beyond hyperparameter-tuning with a correctly-specified evaluation metric, thera re two other methods to boost predictive performance:
# - bagging and 
# - stack-ensembling
# 
# You’ll often see performance improve if you specify `num_bag_folds = 5-10`, `num_stack_levels = 1-3` in the call to `fit()`. Beware that doing this will increase training times and memory/disk usage.
# 
# 

# In[12]:


predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    num_bag_folds=5,
    num_bag_sets=1,
    num_stack_levels=1,
    # last  argument is just for quick demo here, omit it in real applications
    hyperparameters={
        "NN": {"num_epochs": 2},
        "GBM": {"num_boost_round": 20},
    },
)


# You should not provide `tuning_data` when stacking/bagging, and instead provide all your available data as train_data (which AutoGluon will split in more intelligent ways). Parameter `num_bag_sets` controls how many times the K-fold bagging process is repeated to further reduce variance (increasing this may further boost accuracy but will substantially increase training times, inference latency, and memory/disk usage). Rather than manually searching for good bagging/stacking values yourself, AutoGluon will automatically select good values for you if you specify `auto_stack` instead:

# In[13]:


# Folder where to store trained models
save_path = "agModels-predictOccupation"

predictor = TabularPredictor(label=label, eval_metric=metric, path=save_path).fit(
    train_data,
    auto_stack=True,
    time_limit=30,
    # Last 2 arguments are for quick demo, omit them in real applications
    hyperparameters={
        "NN": {"num_epochs": 2},
        "GBM": {"num_boost_round": 20},
    },
)


# Often stacking/bagging will produce superior accuracy than hyperparameter-tuning, but you may try combining both techniques (note: specifying `presets='best_quality'` in `fit()` simply sets `auto_stack=True`).

# ### Prediction options (inference)
# 
# Even if you’ve started a new Python session since last calling `fit()`, you can still load a previously trained predictor from disk:

# In[14]:


# `predictor.path` is another way to get the relative path needed to later load predictor.
predictor = TabularPredictor.load(save_path)


# Above `save_path` is the same folder previously passed to `TabularPredictor`, in which all the trained models have been saved. You can train easily models on one machine and deploy them on another. Simply copy the `save_path` folder to the new machine and specify its new path in `TabularPredictor.load()`.
# 
# We can make a prediction on an individual example rather than on a full dataset:

# In[15]:


# Note: .iloc[0] won't work because it returns pandas Series instead of DataFrame
datapoint = test_data_nolabel.iloc[[0]]

predictor.predict(datapoint)


# To output predicted class probabilities instead of predicted classes, you can use:
# 
# 

# In[16]:


# Returns a DataFrame that shows which probability corresponds to which class
predictor.predict_proba(datapoint)


# By default, `predict()` and `predict_proba()` will utilize the model that AutoGluon thinks is most accurate, which is usually an ensemble of many individual models. Here’s how to see which model this corresponds to:

# In[17]:


predictor.get_model_best()


# We can instead specify a particular model to use for predictions (e.g. to reduce inference latency). Note that a ‘model’ in AutoGluon may refer to for example a single Neural Network, a bagged ensemble of many Neural Network copies trained on different training/validation splits, a weighted ensemble that aggregates the predictions of many other models, or a stacked model that operates on predictions output by other models. This is akin to viewing a RandomForest as one ‘model’ when it is in fact an ensemble of many decision trees.
# 
# Before deciding which model to use, let’s evaluate all of the models AutoGluon has previously trained on our test data:

# ### AutoGluon leaderboard function options

# In[18]:


predictor.leaderboard(test_data, silent=True)


# The leaderboard shows each model’s predictive performance on the test data (`score_test`) and validation data (`score_val`), as well as the time required to: produce predictions for the test data (`pred_time_val`), produce predictions on the validation data (`pred_time_val`), and train only this model (`fit_time`). Below, we show that a leaderboard can be produced without new data (just uses the data previously reserved for validation inside `fit`) and can display extra information about each model:

# In[19]:


predictor.leaderboard(extra_info=True, silent=True)


# The expanded leaderboard shows properties like how many features are used by each model (`num_features`), which other models are ancestors whose predictions are required inputs for each model (`ancestors`), and how much memory each model and all its ancestors would occupy if simultaneously persisted (`memory_size_w_ancestors`). See AutoGluon's leaderboard documentation for full details.
# 
# To show scores for other metrics, you can specify the extra_metrics argument when passing in `test_data`:

# In[20]:


predictor.leaderboard(
    test_data, extra_metrics=["accuracy", "balanced_accuracy", "log_loss"], silent=True
)


# Notice that `log_loss` scores are negative. This is because metrics in AutoGluon are always shown in `higher_is_better` form. This means that metrics such as `log_loss` and `root_mean_squared_error` will have their signs __FLIPPED__, and values will be negative. This is necessary to avoid the user needing to know the metric to understand if higher is better when looking at leaderboard.
# 
# One additional caveat: It is possible that `log_loss` values can be `-inf` when computed via `extra_metrics`. This is because the models were not optimized with `log_loss` in mind during training and may have prediction probabilities giving a class 0 (particularly common with K Nearest Neighbors models). Because `log_loss` gives infinite error when the correct class was given 0 probability, this results in a score of `-inf`. It is therefore recommended that `log_loss` not be used as a secondary metric to determine model quality. Either use `log_loss` as the `eval_metric` or avoid it altogether.

# ### Selecting individual models
# Here’s how to specify a particular model to use for prediction instead of AutoGluon’s default model-choice:

# In[21]:


# index of model to use
i = 0
model_to_use = predictor.get_model_names()[i]
model_pred = predictor.predict(datapoint, model=model_to_use)
print(f"Prediction from {model_to_use} model: {model_pred.iloc[0]}")


# We can easily access information about the trained predictor or a particular model:

# In[22]:


all_models = predictor.get_model_names()
model_to_use = all_models[i]
specific_model = predictor._trainer.load_model(model_to_use)

# Objects defined below are dicts with information (not printed here as they are quite large):
model_info = specific_model.get_info()
predictor_information = predictor.info()


# The predictor also remembers which metric predictions should be evaluated with, which can be done with ground truth labels as follows:

# In[23]:


y_pred_proba = predictor.predict_proba(test_data_nolabel)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)


# Since the label columns remains in the `test_data` DataFrame, we can instead use the shorthand:

# In[24]:


perf = predictor.evaluate(test_data)


# ___
# ## Interpretability: Feature importance
# To better understand our trained predictor, we can estimate the overall importance of each feature:

# In[25]:


predictor.feature_importance(test_data)


# Computed via permutation-shuffling, these feature importance scores quantify the drop in predictive performance (of the already trained predictor) when one column’s values are randomly shuffled across rows. The top features in this list contribute most to AutoGluon’s accuracy (for predicting when/if a patient will be re-admitted to the hospital). Features with non-positive importance score hardly contribute to the predictor’s accuracy, or may even be actively harmful to include in the data (consider removing these features from your data and calling `fit` again). These scores facilitate interpretability of the predictor’s global behavior (which features it relies on for all predictions) rather than local explanations that only rationalize one particular prediction.
# 

# ___
# ## Inference Speed: Model distillation
# 
# While computationally-favorable, single individual models will usually have lower accuracy than weighted/stacked/bagged ensembles. Model Distillation offers one way to retain the computational benefits of a single model, while enjoying some of the accuracy-boost that comes with ensembling. The idea is to train the individual model (which we can call the student) to mimic the predictions of the full stack ensemble (the teacher). Like `refit_full()`, the `distill()` function will produce additional models we can opt to use for prediction.

# ### Training student models

# In[26]:


# Specify much longer time limit in real applications
student_models = predictor.distill(time_limit=30)
student_models


# In[27]:


preds_student = predictor.predict(test_data_nolabel, model=student_models[0])
print(f"predictions from {student_models[0]}: {list(preds_student)[:5]}")


# In[28]:


predictor.leaderboard(test_data, silent=True)


# ### Presets
# 
# If you know inference latency or memory will be an issue, then you can adjust the training process accordingly to ensure `fit()` does not produce unwieldy models.
# 
# One option is to specify more lightweight presets:

# In[29]:


presets = ["good_quality_faster_inference_only_refit", "optimize_for_deployment"]

predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, presets=presets, time_limit=30
)


# ### Lightweight hyperparameters
# Another option is to specify more lightweight hyperparameters:

# In[30]:


predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, hyperparameters="very_light", time_limit=30
)


# Here you can set hyperparameters to either `'light'`, `'very_light'`, or `'toy'` to obtain progressively smaller (but less accurate) models and predictors. Advanced users may instead try manually specifying particular models’ hyperparameters in order to make them faster/smaller.

# ### Excluding models
# 
# Finally, you may also exclude specific unwieldy models from being trained at all. Below we exclude models that tend to be slower (K Nearest Neighbors, Neural Network, models with custom larger-than-default hyperparameters):

# In[31]:


excluded_model_types = ["KNN", "NN", "custom"]
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, excluded_model_types=excluded_model_types, time_limit=30
)


# <p style="padding: 10px; border: 1px solid black;">
# <img src=".././images/MLU-NEW-logo.png" alt="drawing" width="400"/> <br/>
# 
# # Thank you!

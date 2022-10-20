import numpy as np
import torch
import joblib
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
StratifiedShuffleSplit
from .. import helpers
from copy import deepcopy

class Pipe():
    
    def __init__(self, *pipeline, pipeline_fileloc=None):
        """
        Create a Pipe from a Pipeline object or a pickled Pipeline object
        """
        if pipeline and (type(pipeline) is list or type(pipeline) is tuple):
            self.pipeline = make_pipeline(*pipeline)
        elif pipeline:
            self.pipeline = pipeline
        else:
            self.pipeline = self.load(pipeline_fileloc)
    
    def save(self, pipeline_fileloc):
        """
        Save the Pipe as a pickled file
        """
        joblib.dump(self.pipeline, pipeline_fileloc)
        
    def load(self, pipeline_fileloc):
        """
        Load a Pipe from a pickled file
        """
        return joblib.load(pipeline_fileloc)


def fit_pipe(feat_train, labels_train, preproc_init, classify_init, preproc_refit=True):
    """
    Fit a full pipeline, either maintaining the initialized Preprocessor or refitting it
    
    JZ 2022
    
    Args:
        feat_train (np.array):
         Features on which to train the model
        labels_train (np.array):
         Labels on which to train the model
        preproc_init (sklearn Transform):
         Preprocessor pipeline for features training data
        classify_init (sklearn Classifier):
         Classifier (or pipeline) for features training data
        preproc_refit (bool):
         Whether or not to refit the preprocessor to the training data
         before constructing the final pipeline
    
    Returns:
        Fitted Pipeline
    """
    
    preproc = deepcopy(preproc_init)
    if preproc_refit:
        preproc.fit(feat_train, labels_train)

    classify = deepcopy(classify_init)
    classify.fit(preproc.transform(feat_train), labels_train)
    pipe = Pipe(preproc, classify)
    
    return pipe

def stratified_sample(features_train, labels_train, n_splits=1, train_size=None, test_size=None):
    assert (train_size is not None or test_size is not None) and not (train_size is None and test_size is None), "JZ Error: Exactly one of train_size and test_size should be specified"
    
    if train_size is not None:
        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size)
    else:
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    return list(sss.split(features_train, labels_train))

def fit_n_train(features_train, labels_train, preproc_init, classify_init, preproc_refit=True, n_train=1e1):
    """
    Fit a full pipeline, using only of n_train datapoints from features_train/labels_train
    
    JZ 2022
    
    Args:
        feat_train (np.array):
         Features on which to train the model
        labels_train (np.array):
         Labels on which to train the model
        preproc_init (sklearn Transform):
         Preprocessor pipeline for features training data
        classify_init (sklearn Classifier):
         Classifier (or pipeline) for features training data
        preproc_refit (bool):
         Whether or not to refit the preprocessor to the training data
         before constructing the final pipeline
        n_train (int):
         Number of training examples on which to fit the model
         
    Returns:
        Fitted Pipeline
    """
    
    if n_train < features_train.shape[0]:
        train_size = n_train/features_train.shape[0]
        train_subset_inx, _ = stratified_sample(features_train, labels_train, n_splits=1, train_size=train_size)[0]
    else:
        n_train = features_train.shape[0]
        train_subset_inx = list(range(n_train))
    
#     print(train_size, sss, len(train_subset_inx), train_subset_inx)
    
    features_train_subset, labels_train_subset = features_train[train_subset_inx], labels_train[train_subset_inx]
    pipe = fit_pipe(features_train_subset, labels_train_subset, preproc_init, classify_init, preproc_refit=preproc_refit)
    
    return pipe, n_train
    
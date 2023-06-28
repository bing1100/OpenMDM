import pandas as pd
from sklearn import tree, linear_model, ensemble, naive_bayes, discriminant_analysis, svm
import numpy as np
import math
import mlserialize as s
import json
from recordlinkage.classifiers import ECMClassifier
from sklearn import base

instantiate = lambda key, **kwargs: {
    "LogisticRegression":MDMLogisticRegression(**kwargs),
    "DecisionTreeClassifier":MDMDecisionTree(**kwargs),
    "RandomForestClassifier":MDMRandomForest(**kwargs),
    "GaussianNaiveBayes":MDMGaussianNB(**kwargs),
    "Perceptron":MDMPerceptron(**kwargs),
    "GradientBoostingClassifier":MDMGradientBoosting(**kwargs),
    "MDMECM":MDMECM(**kwargs),
    "VotingClassifier":MDMVoting(**kwargs)
}.get(key, MDMNode(**kwargs))

class LinearClassifierMixin(base.ClassifierMixin):
    """Mixin for linear classifiers.
    Handles prediction for sparse and dense X.
    """

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the confidence scores.
        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per `(n_samples, n_classes)` combination. In the
            binary case, confidence score for `self.classes_[1]` where >0 means
            this class would be predicted.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(X)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return xp.reshape(scores, -1) if scores.shape[1] == 1 else scores

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        xp, _ = get_namespace(X)
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = xp.astype(scores > 0, int)
        else:
            indices = xp.argmax(scores, axis=1)

        return xp.take(self.classes_, indices, axis=0)

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.
        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

def mask_data(xs, include):
    return [[x[i] for i in include] for x in xs] 

def genMatchField(name, algoName, algoType, pathName="resourcePath", resourcePath=None, thresh=None, filename=None):
    mlJson = {
        "name": f"{name}",
        algoType: {
            "algorithm": f"{algoName}",
        }
    }
        
    if resourcePath:
        mlJson[pathName] = resourcePath
    
    if filename:
        mlJson[algoType]["savePath"] = filename
    
    if thresh:
        mlJson[algoType]["matchThreshold"] = float(thresh)
    
    return mlJson

def get_model_data(filename):
    infile = open(filename, 'r')
    datum = json.load(infile)
    return datum

def dump_model(filename, model_name, data):
    nfilename = filename + "-" + model_name +".json"
    out_file = open(nfilename, "w")
    json.dump(data, out_file)
    return nfilename

class MDMNode:
    id = 0
    def __init__(self, idx=0, thresh=0.5, algo="root", **kwargs):
        self.id = MDMNode.id
        MDMNode.id += 1
        self.algo = algo
        self.x_output = []
        self.y_output = []
        self.is_end = False
        self.children = {}
        
        self.cmp = lambda x : x[idx] > thresh
        
    def instantiate(self, xs, ys):
        # Check if Root Node
        self.x_output = xs
        self.y_output = ys
        
        if self.algo == "root":
            return self.x_output, self.y_output
        
        for i, (x, y) in enumerate(zip(xs, ys)):
            if y != 1:
                continue
            res = self.cmp(x)
            if not res:
               self.y_output[i] = 0

        return self.x_output, self.y_output
    
    def _predict_train(self, xs):
        y_output = [1]*len(xs)
        
        if self.algo == "root":
            return y_output
        
        for i, (x, y) in enumerate(zip(xs, y_output)):
            if y != 1:
                continue
            res = self.cmp(x)
            if not res:
               y_output[i] = 0

        return y_output
    
    def predict(self, xs):
        pred = self._predict_train(xs)
        return pred, pred
    
    def fit(self, x_train, y_train):
        pass
    
    def from_checkpoint(self, mdmjson):
        pass
        
    def save(self, filename):
        return None, None, None
    
    def load(self, filename):
        pass

class MDMDecisionTree(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = tree.DecisionTreeClassifier()
        self.thresh = kwargs.get("thresh", 0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches

    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("DecisionTreeClassifier")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_decision_tree(model)
        nfilename = dump_model(filename, "DecTre", datum)

        matchfields = [genMatchField("DecTre", "DecisionTreeClassifier", "ml", filename=nfilename)]
        matchresultmap = {
            "DecTre":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_decision_tree(datum)
        
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)
    
class MDMLogisticRegression(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = linear_model.LogisticRegression(
            penalty='l2',
            solver='newton-cholesky'
        )
        self.thresh = kwargs.get("thresh", 0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches

    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("LogisticRegression")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_logistic_regression(model)
        nfilename = dump_model(filename, "LogReg", datum)
        matchfields = [genMatchField("LogReg", "LogisticRegression", "ml", filename=nfilename)]
        matchresultmap = {
            "LogReg":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_logistic_regression(datum)
        
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMRandomForest(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = ensemble.RandomForestClassifier(
            min_samples_leaf=10,
        )
        self.thresh = kwargs.get("thresh", 0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches

    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("RandomForestClassifier")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_random_forest(model)
        nfilename = dump_model(filename, "RanFor", datum)
        matchfields = [genMatchField("RanFor", "RandomForestClassifier", "ml", filename=nfilename)]
        matchresultmap = {
            "RanFor":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_logistic_regression(datum)
        
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMGaussianNB(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = naive_bayes.GaussianNB()
        self.thresh = kwargs.get("thresh",0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches
    
    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("GaussianNaiveBayes")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_gaussian_nb(model)
        nfilename = dump_model(filename, "GausNB", datum)
        matchfields = [genMatchField("GausNB", "GaussianNaiveBayes", "ml", filename=nfilename)]
        matchresultmap = {
            "GausNB":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_gaussian_nb(datum)
    
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMPerceptron(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = linear_model.Perceptron(
            penalty='elasticnet',
            early_stopping=True,
        )
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict(xs)
    
    def predict(self, xs, ys=None):
        pred = self.model.predict(xs)
        return pred, pred
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("Perceptron")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_perceptron(model)
        nfilename = dump_model(filename, "Percep", datum)
        matchfields = [genMatchField("Percep", "Perceptron", "ml", filename=nfilename)]
        matchresultmap = {
            "Percep":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_perceptron(datum)
        
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)


class MDMGradientBoosting(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = ensemble.GradientBoostingClassifier(
            min_samples_leaf=10,
            n_iter_no_change=3,
        )
        
        self.thresh = kwargs.get("thresh",0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches

    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("GradientBoostingClassifier")
        self.load(filename)
    
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_gradient_boosting(model)
        nfilename = dump_model(filename, "GraBoo", datum)
        matchfields = [genMatchField("GraBoo", "GradientBoostingClassifier", "ml", filename=nfilename)]
        matchresultmap = {
            "GraBoo":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_gradient_boosting(datum)
    
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMECM(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = ECMClassifier()

    def fit(self, x_train: pd.DataFrame):
        self.model.fit(x_train)

    def _predict_train(self, xs, ys=None):
        return self.model.predict(xs)
    
    def predict(self, xs, ys=None):
        pred = self.model.predict(xs)
        return pred, pred

    def save(self, filename, model=None):
        if not model:
            model=self.model
        data = {
            'feature_log_prob': model.feature_log_prob_.tolist(),
            'feature_labels': model._column_labels.tolist(),
            'class_log_prior': model.class_log_prior_.tolist(),
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.model.feature_log_prob_ = np.array(data['feature_log_prob'])
        self.model._column_labels = np.array(data['feature_labels'])
        self.model.class_log_prior_ = np.array(data['class_log_prior'])

class MDMVoting(LinearClassifierMixin):
    def __init__(self, **kwargs):
        if "matcher" in kwargs:
            matcher = kwargs.get("matcher")
            if "classifiers" in matcher:
                self.estimators = []
                for clfname in matcher.get("classifiers"):
                    self.estimators.append((clfname, instantiate(clfname)))
                
                self.model = ensemble.VotingClassifier(estimators=self.estimators, voting='soft')
                self.thresh = kwargs.get("thresh",0.5)
                self.pthresh = kwargs.get("pthresh", 0.5)
                self.tthresh = (self.thresh + self.pthresh)/2
                
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches

    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filenames = mdmjson.get_checkpoint("VotingClassifier")
        self.load(filenames)
        
    def save(self, filename):
        filenames = []
        models = self.model.named_estimators_
        for clfname, clf in self.estimators:
            _, _, matchfields = clf.save(filename, model=models[clfname].model)
            nfilename = matchfields[0]["ml"]["savePath"]
            filenames.append((clfname, nfilename))
        matchfields = [genMatchField("VotCla", "VotingClassifier", "ml", filename=filenames)]
        matchresultmap = {
            "VotCla":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filenames):
        for clfname, filename in filenames:
            self.estimators.append((clfname, instantiate(clfname)))
            self.estimators[-1][1].load(filename)
        
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)
        
class MDMBernoulliNB(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = naive_bayes.BernNB()
        self.thresh = kwargs.get("thresh",0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches
    
    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("BernoulliNaiveBayes")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_bernoulli_nb(model)
        nfilename = dump_model(filename, "BernNB", datum)
        matchfields = [genMatchField("BernNB", "BernoulliNaiveBayes", filename=nfilename)]
        matchresultmap = {
            "BernNB":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_bernoulli_nb(datum)
    
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMMultinomialNB(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = naive_bayes.MultinomialNB()
        self.thresh = kwargs.get("thresh",0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches
    
    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("MultinomialNaiveBayes")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_multinomial_nb(model)
        nfilename = dump_model(filename, "MultNB", datum)
        matchfields = [genMatchField("MultNB", "MultinomialNaiveBayes", filename=nfilename)]
        matchresultmap = {
            "MultNB":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_multinomial_nb(datum)
    
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMComplementNB(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = naive_bayes.ComplementNB()
        self.thresh = kwargs.get("thresh",0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches
    
    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("ComplementNaiveBayes")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_complement_nb(model)
        nfilename = dump_model(filename, "CompNB", datum)
        matchfields = [genMatchField("CompNB", "ComplementNaiveBayes", filename=nfilename)]
        matchresultmap = {
            "CompNB":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_complement_nb(datum)
    
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMLDA(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = discriminant_analysis.LinearDiscriminantAnalysis()
        self.thresh = kwargs.get("thresh",0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches
    
    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("LinearDiscriminantAnalysis")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_lda(model)
        nfilename = dump_model(filename, "LDA", datum)
        matchfields = [genMatchField("LDA", "LinearDiscriminantAnalysis", filename=nfilename)]
        matchresultmap = {
            "LDA":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_lda(datum)
    
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMQDA(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = discriminant_analysis.QuadraticDiscriminantAnalysis()
        self.thresh = kwargs.get("thresh",0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches
    
    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("QuadraticDiscriminantAnalysis")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_qda(model)
        nfilename = dump_model(filename, "QDA", datum)
        matchfields = [genMatchField("QDA", "QuadraticDiscriminantAnalysis", filename=nfilename)]
        matchresultmap = {
            "QDA":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_qda(datum)
    
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

class MDMSVM(LinearClassifierMixin):
    def __init__(self, **kwargs):
        self.model = svm.SVC(
            probability=True,
        )
        self.thresh = kwargs.get("thresh",0.5)
        self.pthresh = kwargs.get("pthresh", 0.5)
        self.tthresh = (self.thresh + self.pthresh)/2
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def _predict_train(self, xs, ys=None):
        return self.model.predict_proba(xs)[:,1] > self.tthresh
    
    def predict(self, xs, ys=None):
        matches = self.model.predict_proba(xs)[:,1] > self.thresh
        pmatches = np.all([self.pthresh < self.model.predict_proba(xs)[:,1],
                          self.model.predict_proba(xs)[:,1] < self.thresh],
                          axis=0)
        return matches, pmatches
    
    def predict_proba(self, xs, ys=None):
        return self.model.predict_proba(xs)
    
    def from_checkpoint(self, mdmjson):
        filename = mdmjson.get_checkpoint("SupportVectorMachine")
        self.load(filename)
        
    def save(self, filename, model=None):
        if not model:
            model=self.model
        datum = s.serialize_svm(model)
        nfilename = dump_model(filename, "SVM", datum)
        matchfields = [genMatchField("SVM", "SupportVectorMachine", filename=nfilename)]
        matchresultmap = {
            "SVM":"MATCH"
        }
        return "MLMatchResultMap", matchresultmap, matchfields
    
    def load(self, filename):
        datum = get_model_data(filename)
        self.model = s.deserialize_svm(datum)
    
    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)


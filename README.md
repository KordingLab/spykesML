# SpykesML
## Machine Learning methods implemented as encoding models
This repository accompanies "Modern Machine Learning Far Outperforms GLMs at Predicting Spikes"[https://doi.org/10.1101/111450]. Here, you can find a Python class `MLencoding`
that you can use for quickly making encoding models.

For a tutorial on how to use `MLencoding`, see the notebooks folder. There
you can also find a "standalone notebook" that demos how to use some ML methods
without our fancy class.

Currently implemented methods:
* GLM
* 2-layer feedforward net
* Random forest
* xgboost
* LSTM


#### Installation
In a terminal:
```
git clone https://github.com/KordingLab/spykesML
cd spykesML
python setup.py install
```

#### Quick how-to:
Build the encoder:
```python
model = MLencoding(tunemodel = 'xgboost')
print(model.params)
```
Fit and predict to some data:
```python
model.fit(X_train, y_train)
Y_hat = model.predict(X_test)
```
Perform k-fold cross-validation:
```python
model.fit_cv(X,y)
```
Use spike and covariate history as inputs:
```python
xgb_history = MLencoding(tunemodel = 'xgboost',
                         cov_history = True, spike_history=True,
                         window = 50, #this dataset has 50ms time bins
                         n_filters = 4,
                         max_time = 250 ) #in ms
xgb_history.fit_cv(X,y, verbose = 2, continuous_folds = True)
```
See the tutorial for how to define parameters, build new encoding models, and more!

### Dependencies
#### Basics
 - numpy
 - pandas
 - scipy
 - matplotlib

#### Methods
 - sklearn
 - pyglmnet (glm)
 - xgboost
 - theano (NN)
 - keras (NN)


![predictfirst](https://github.com/KordingLab/spykesML/blob/master/predictfirst.jpg)

import numpy as np
import warnings

#GLM
from pyglmnet import GLM

#NN
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.regularizers import l2
from keras.optimizers import Nadam, adam
from keras.layers.normalization import BatchNormalization

#CV
from sklearn.model_selection import KFold

#XGB
import xgboost as xgb

#RF
from sklearn.ensemble import RandomForestRegressor

#LSTM
from keras.layers import  LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda

class MLencoding(object):
    """
    This class implements several conveniences for fitting and
    and predicting a neuron's activity with encoding models built
    from machine learning methods.

    Parameters
    ----------
    tunemodel: str, {'glm','feedforward_nn','xgboost','lstm','random_forest'}
                    OR
                sklearn-style instance with methods 'fit' & 'predict'
    random_state: int, seed for numpy.random`
    verbose: bool, whether to print convergence / loss, default: False

    History terms (optional):
    cov_history = whether to use covariate history in prediction
    spike_history = whether to use spike history in predicition
    max_time = how long back, in ms, to use covariate/spike history from (if using)
    n_filters = number of temporal bases to use to span the interval [0, max_time]
    window = how many ms are in each time bin?
    n_every = predict all time bins (nevery = 1), or every once in a while?
                Note that if time bins are randomly selected for test/train split,
                use nevery >= max_time/time_window to prevent leakage


    For ensemble fitting:
        Fit an stacked ensemble by setting is_ensemble = True. This requires that you
        supply a list of 1st-order methods.
    is_ensemble = True/False. whether to train tunemodel on the results of 1st-order models
    first_order_models = list of MLenconding instances. Needed if is_ensemble = True



    Callable methods
    ----------------
    set_params
    fit
    fit_cv
    predict
    get_params

    Internal methods (also Callable)
    -------------------------------

    get_all_with_history_keras
    poisson_pseudoR2
    raised_cosine_filter
    temporal_filter


    """

    def __init__(self, tunemodel='glm', spike_history = False, cov_history = False,
                 random_state=1, window = 0, n_filters = 0, max_time = 0, n_every = 1,
                 verbose=0, is_ensemble = False, first_order_models = None):
        """
        Initialize the object
        """


        self.tunemodel = tunemodel
        self.verbose = verbose
        self.spike_history = spike_history
        self.cov_history = cov_history
        self.window = window
        self.n_filters = n_filters
        self.max_time = max_time
        self.n_every = n_every
        self.is_ensemble = is_ensemble
        self.first_order_models = first_order_models

        np.random.seed(random_state)

        if isinstance(tunemodel,str):
            valid_models = ['glm','feedforward_nn','xgboost','lstm','random_forest']
            if tunemodel not in valid_models:
                raise NotImplementedError('Invalid model type. Got {}, not in {}'.format(
                                    tunemodel, valid_models))

            # Assign optimization parameters
            # -------------------------------
            self.set_default_params()


        # If tunemodel is not a str we assume it's a predefined sklearn-style model
        else:
            self.model = tunemodel

        if spike_history or cov_history:
            try:
                assert all([p>0 for p in [window, n_filters, max_time]])
                assert isinstance(n_filters, int)

            except AssertionError:
                print('window,n_filters, and max_time must all be ' +
                    ' greater than 0 if spike or covariate history are used.')
                raise
            try:
                assert int(max_time/window) >= n_filters
            except AssertionError:
                print('There are more time filters than there are time points '+
                        'per example. Need max_time//window >= n_filters')
                raise

        if tunemodel == 'lstm':
            try:
                assert spike_history and cov_history
            except AssertionError:
                print('Recurrent models need history!\n' +
                    'Set spike_history=True and cov_history=True')
                raise

        if is_ensemble == True:
            try:
                for model in first_order_models:
                    assert isinstance(model, MLencoding)
            except:
                print('first_order_models needs to be a list of MLencoding objects '+
                        'if is_ensemble == True.')
                raise

    def set_default_params(self):
        """
        A function that sets model parameters to some default.

        """
        tunemodel = self.tunemodel

        # Assign 'default' parameters;
        if tunemodel == 'glm':
            self.params = {'distr':'softplus', 'alpha':0.1, 'tol':1e-8,
              'reg_lambda':np.logspace(np.log(0.05), np.log(0.0001), 10, base=np.exp(1)),
              'learning_rate':2, 'max_iter':10000, 'eta':2.0}
        elif tunemodel == 'feedforward_nn':
            self.params = {'dropout': 0.05,
              'l2': 1.6e-08,
              'lr': 0.001,
              'n1': 76, #number of layers in 1st hidden layer
              'n2': 16,
              'decay': 0.009, 'clipnorm' : 1.3, 'b1' : 0.2, 'b2' : 0.02}

        elif tunemodel == 'xgboost':
            self.params = {'objective': "count:poisson", #for poisson output
                'eval_metric': "logloss", #loglikelihood loss
                'seed': 2925, #for reproducibility
                'silent': 1,
                'learning_rate': 0.05,
                'min_child_weight': 2, 'n_estimators': 580,
                'subsample': 0.6, 'max_depth': 5, 'gamma': 0.4}
        elif tunemodel == 'random_forest':
            self.params = {'max_depth': 15,
             'min_samples_leaf': 4,
             'min_samples_split': 5,
             'min_weight_fraction_leaf': 0.0,
             'n_estimators': 471}
        elif tunemodel == 'lstm':
             self.params = {'epochs': 8, 'n_units': 45, 'dropout': 0.00491871366927632,
                              'batch_size': 101}

        if isinstance(tunemodel,str):
            self.default_params = True

    def set_params(self,params):
        """Method for setting the parameters of the regression method."""
        assert isinstance(params,dict)

        for k in params.keys():
            self.params[k] = params[k]

        if not isinstance(self.tunemodel,str):
            # can use this method to access sklearn's set_params method
            # if method predefined and not a string
            self.tunemodel.set_params(**params)

        self.default_params = False

    def get_params(self):
        """Prints the current parameters of the model."""
        return self.params

    def fit(self, X, Y, get_history_terms = True):
        """
        Fits the model to the data in X to predict the response Y.

        Imports models and creates model instance as well.

        Parameters
        ----------
        X: float, n_samples x n_features, features of interest
        Y: float, n_samples x 1, population activity
        get_history_terms = Boolean. Whether to compute the temporal features.
                    Note that if spike_history and cov_history are False,
                    no history will be computed anyways and the flag does nothing.


        """
        if self.default_params:
            warnings.warn('\n  Using default hyperparameters. Consider optimizing on'+
                ' a held-out dataset using, e.g. hyperopt or random search')

        # make the covariate matrix. Include spike or covariate history?
        # The different methods here are to satisfy the needs of recurrent keras
        # models
        if get_history_terms:
            if self.tunemodel == 'lstm':
                X, Y = self.get_all_with_history_keras(X, Y)
            else:
                X, Y = self.get_all_with_history(X, Y)

        if self.tunemodel == 'glm':
            model = GLM(**self.params)
            model.fit(X, Y)

            # we want the last of the regularization path
            self.model = model[-1]

        elif self.tunemodel == 'feedforward_nn':

            if np.ndim(X)==1:
                X = np.transpose(np.atleast_2d(X))

            params = self.params
            model = Sequential()
            model.add(Dense(params['n1'], input_dim=np.shape(X)[1], kernel_initializer='glorot_normal',
                        activation='relu', kernel_regularizer=l2(params['l2'])))
            model.add(Dropout(params['dropout']))
            model.add(BatchNormalization())
            model.add(Dense(params['n2'], kernel_initializer='glorot_normal'
                            , activation='relu',kernel_regularizer=l2(params['l2'])))
            model.add(BatchNormalization())
            model.add(Dense(1,activation='softplus'))
            optim = adam(lr=params['lr'], clipnorm=params['clipnorm'],
                            decay = params['decay'],
                            beta_1=1-params['b1'], beta_2=1-params['b2'])
            model.compile(loss='poisson', optimizer=optim,)
            hist = model.fit(X, Y, batch_size = 128, epochs=30, verbose=self.verbose)

            self.model = model

        elif self.tunemodel == 'xgboost':

            dtrain = xgb.DMatrix(X, label=Y)
            num_round = 200
            self.model = xgb.train(self.params, dtrain, num_round)

        elif self.tunemodel == 'random_forest':

            self.model = RandomForestRegressor(**self.params)
            self.model.fit(X, Y)

        elif self.tunemodel == 'lstm':

            if np.ndim(X)==1:
                X = np.transpose(np.atleast_2d(X))

            params = self.params
            model=Sequential() #Declare model
            #Add recurrent layer
            model.add(LSTM(int(params['n_units']),input_shape=(X.shape[1],X.shape[2]),\
                           dropout_W=params['dropout'],dropout_U=params['dropout']))
                            #Within recurrent layer, include dropout
            model.add(Dropout(params['dropout'])) #Dropout some units (recurrent layer output units)

            #Add dense connections to output layer
            model.add(Dense(1,activation='softplus'))

            #Fit model (and set fitting parameters)
            model.compile(loss='poisson',optimizer='rmsprop',metrics=['accuracy'])
            model.fit(X,Y,epochs=int(params['epochs']),
                        batch_size = int(params['batch_size']),verbose=self.verbose) #Fit the model

            self.model = model

        else: #using predefined model
            self.model.fit(X,Y)

    def predict(self, X, get_history_terms = True):
        """
        Compute the firing rates for the neuron
        based on the fit of specified tuning model.


        Parameters
        ----------
        X: float, n_samples x n_features, feature of interest
        get_history_terms = Boolean. Whether to compute the temporal features.
                    Note that if spike_history and cov_history are False,
                    no history will be computed anyways and the flag does nothing.

        Outputs
        -------
        Y: float, (n_samples,) , predicted activity
        """
        Y_null = np.zeros((X.shape[0],))


        if get_history_terms:
            if self.tunemodel == 'lstm':
                X, Y_null = self.get_all_with_history_keras(X, Y_null)
            else:
                X, Y_null = self.get_all_with_history(X, Y_null)


        if self.tunemodel == 'xgboost':
            X = xgb.DMatrix(X)

        Y = self.model.predict(X)

        return Y.flatten()



    def raised_cosine_filter(self,n_bins, k, nBases = 15):
        """Return a cosine bump, kth of nBases, such that the bases tile
        the interval [0, n_bins].

        To plot these bases:
        for i in range(10):
            b =  raised_cosine_filter(250, i, nBases = 10)
            plt.plot(b)
        """
        assert all([isinstance(p,int) for p in [n_bins, k, nBases]])

        t = np.linspace(0,self.max_time,n_bins)

        nt = np.log(t+0.1)

        cSt,cEnd = nt[1],nt[-1]
        db = (cEnd - cSt) / (nBases)
        c = np.arange(cSt,cEnd,db)

        bas = np.zeros((nBases,t.shape[0]))

        filt = lambda x: ( np.cos( \
                np.maximum(-np.pi, np.minimum(np.pi,(nt - c[x])*np.pi/(db) )) ) \
                + 1) / 2;

        this_filt = filt(k)

        return this_filt/np.sum(this_filt)

    def temporal_filter(self,variables,nfilt=10, keras_format = False, scale = None):
        """ Performs convolution of various filters upon each variable (column) in the input array
        Inputs:
        variables = an array of shape (n_bins, n_variables)
        nfilt = number of filters
        keras_format = return a 2d or 3d array
        scale = function for scaling, centering variables.

        Outputs:
        history_filters = an array of shape(n_bins, n_variables x n_filters)
                                            OR
                          an array of shape(n_bins, n_filters, n_variables) if keras_format
        ^ these are different shapes because of the redundant 3D
        format that Keras wants its variables for RNNs
        """
        if scale == None:
            scale = lambda x: x

        if variables.ndim == 1:
            variables = np.reshape(variables,(variables.shape[0],1))
        # We'll use 10 bases up to 250 ms
        window = self.window

        n_bins = int(self.max_time/window)
        n_vars = variables.shape[1]

        history_filters = np.zeros((variables.shape[0],n_vars*nfilt))
        if keras_format:
            history_filters = np.zeros((variables.shape[0],nfilt,n_vars))


        for i in range(nfilt):
            #get raised cosine filter
            filt = self.raised_cosine_filter(n_bins,i,nfilt)
            #apply it to each variable
            this_filter = np.zeros(variables.shape)
            for j in range(n_vars):
                temp = np.convolve(variables[:,j],filt)[:variables.shape[0]]
                this_filter[:,j] = temp

            if keras_format:
                history_filters[:, i, :] = scale(this_filter)
            else:
                history_filters[:,(n_vars*i):(n_vars*(i+1))] = this_filter

        return history_filters

    def get_all_with_history(self,raw_covariates, raw_spikes,
                        cov_history = None,
                        nfilt = None, spike_history = None,
                        normalize = False,
                        n_every = None):
        """

        Inputs:
        raw_spikes = (nbins,) array of binned spikes
        raw_covariates = (nbins,nvars) array of binned covariates
        cov_history = whether to use covariate history in prediction
        spike_history = whether to use spike history in predicition
        normalize = whether to set normalize mean and covariance of all covariates & their history
        n_every = predict all time bins (nevery = 1), or every once in a while?
                    Note that if time bins are randomly selected for test/train split,
                    use nevery >= 250/time_window to prevent leakage
        nfilt = number of temporal features. Uses raised cosine bases up to 250 ms



        Returns:
        covariates = array with columns as covariates. Columns go:
                [current cov.] + [cov. convolved with temporal filters, if chosen] +
                [spike history convolated with filters]
        spikes = an array of spike bins, to be used as training/test Y

        """
        if cov_history == None:
            cov_history = self.cov_history
        if nfilt == None:
            nfilt = self.n_filters
        if spike_history == None:
            spike_history = self.spike_history
        if n_every == None:
            n_every = self.n_every

        assert raw_spikes.ndim == 1

        data_indices = range(n_every-1,raw_spikes.shape[0],n_every)

        spikes = raw_spikes[data_indices]
        covariates = raw_covariates[data_indices,:]


        # then we convolve spikes to get spike histories
            # will be (n_bins, nfilt) array

        if spike_history:
            spike_histories = self.temporal_filter(raw_spikes,nfilt)
            assert spike_histories.shape == (raw_spikes.shape[0],nfilt)
            covariates = np.hstack((covariates,spike_histories[data_indices,:]))

        # and we get covariate histories
        if cov_history:
            cov_histories = self.temporal_filter(raw_covariates,nfilt)
            assert cov_histories.shape == (raw_spikes.shape[0],nfilt*raw_covariates.shape[1])
            covariates = np.hstack((covariates,cov_histories[data_indices,:]))

        if normalize:
            from sklearn.preprocessing import scale
            covariates = scale(covariates)

        return covariates, spikes

    def get_all_with_history_keras(self, raw_covariates,raw_spikes,
                        bins_before=0,temporal_bases = None,
                            spike_history= None,
                             covariate_history = None, normalize = True):
        """
        Function that creates the covariate matrix for a Keras LSTM (or RNN, etc.)
        Note: assumes continuity of data. Call on separate CV folds
        Note: covariate_history must be true, otherwise LSTM doesn't really make sense

        ----------
        raw_spikes: a matrix of shape (n_samples,)
            the number of spikes in each time bin for each neuron
        raw_covariates: a matrix of size "number of time bins" x "number of covariates"
            the number of spikes in each time bin for each neuron

        temporal_bases: None, or int
            Whether to use bins or a convolved kernal with some number of features
             If no temporal bases, would you like to use the raw bins before? -->
        bins_before: integer
            How many bins of neural data prior to the output are used
            Ignored if temporal_bases is > 0

        Returns
        -------
        X: a matrix of size "number of total time bins - number of temporal items"
                                x "number of temporal items" x "1+n_features"
        """
        if temporal_bases==None:
            temporal_bases = self.n_filters
        if spike_history==None:
            spike_history= self.spike_history
        if covariate_history==None:
            covariate_history = self.cov_history



        assert raw_spikes.ndim == 1 and raw_covariates.shape[0]==raw_spikes.shape[0]
        assert covariate_history #

        num_examples=raw_spikes.shape[0] #Number of total time bins we have neural data for
        n_features = raw_covariates.shape[1]

        sh = ch = 0
        if spike_history: sh = 1
        if covariate_history: ch = 1

        if normalize:
            from sklearn.preprocessing import scale
            raw_covariates = scale(raw_covariates)
        else: scale = lambda x: x

        if temporal_bases:

            first_n = int(self.max_time/self.window) # early bins where we don't have all covariates

            spikes = raw_spikes[:]
            covariates = np.zeros((num_examples, 1+temporal_bases, sh+ch*n_features))
            covariates[:,0,sh:] = raw_covariates # input current covariates

            # then we convolve spikes to get spike histories

            if spike_history:
                spike_histories = self.temporal_filter(raw_spikes,temporal_bases,
                                        scale=scale)
                assert spike_histories.shape == (num_examples,temporal_bases)
                covariates[:,1:,0] = spike_histories # spike history input will be 0 at 'curr' input

            # and we get covariate histories
            if covariate_history:
                cov_histories = self.temporal_filter(raw_covariates,
                            temporal_bases, keras_format = True, scale=scale)
                assert cov_histories.shape == (num_examples,temporal_bases,n_features)
                covariates[:,1:,sh:] = cov_histories

            #drop incomplete samples
            covariates = covariates[:,:,:]

        elif bins_before:
            # This part adapted from Josh Glaser's code

            spikes = raw_spikes[:]
            covariates = np.zeros((num_examples, 1+bins_before, sh+ch*n_features))


            covariates[:,0,sh:] = raw_covariates # input current covariates

            #Loop through each time bin, and collect the spikes occurring in surrounding time bins
            #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
            #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8


            for start_idx in range(num_examples-bins_before): #The first bins_before
                    #The bins of neural data we will be including are between start_idx
                    #and end_idx (which will have length "bins_before")
                end_idx=start_idx+bins_before;


                if spike_history:
                    #Put neural data from surrounding bins in X, starting at row "bins_before"
                    covariates[start_idx+bins_before,1:,0]=raw_spikes[start_idx:end_idx]

                if covariate_history:
                    #Put neural data from surrounding bins in X, starting at row "bins_before"
                    covariates[start_idx+bins_before,1:,sh:]=raw_covariates[start_idx:end_idx,:]

            #drop incomplete samples
            covariates = covariates[:,:,:]
        else:
            covariates, spikes = raw_covariates, raw_spikes

        return covariates, spikes

    def fit_cv(self, X, Y, n_cv=10, verbose=1, continuous_folds = False):
        """Performs cross-validated fitting.

        Input
        =====
        X  = input data
        Y = spiking data
        n_cv = number of cross-validations folds
        continuous_folds = True/False. whether to split folds randomly or to
                        split them in contiguous chunks. The latter is advantageous
                        when using spike history as a covariate to prevent
                        leakage across folds

        Returns
        (Y_hat, pR2_cv); a vector of predictions Y_hat with the
        same dimensions as Y, and a list of pR2 scores on each fold pR2_cv.





        """
        if not continuous_folds:
            if self.spike_history:
                try:
                    assert self.n_every >= int(self.max_time/self.window)
                except AssertionError:
                    print('Warning: Using random CV folds when spike history is used ' + \
                    'will cause data leakage unless we predict spikes at an ' + \
                    'interval greater than the length of history used to predict.\n'+\
                    'Set continuous_folds = True '+ \
                    'or increase n_every above max_time/window' )

        if self.tunemodel=='lstm':
            assert continuous_folds

        if self.is_ensemble == True:
            print('Running nested CV scheme on first order models.')
            return self.ensemble_cv(X, Y, continuous_folds = continuous_folds,
                                        n_cv_outer=n_cv, n_cv_inner=n_cv)

        if np.ndim(X)==1:
            X = np.transpose(np.atleast_2d(X))

        n_samples = X.shape[0]

        # get history terms
        if self.tunemodel == 'lstm':
            X, Y = self.get_all_with_history_keras(X, Y)
        else:
            X, Y = self.get_all_with_history(X, Y)

        Y_hat=np.zeros(len(Y))
        pR2_cv = list()

        if continuous_folds:
            # sporadic prediction with continuous_folds not yet implemented
            assert self.n_every==1

            for i in range(n_cv):
                if verbose > 1:
                    print( '...runnning cv-fold', i, 'of', n_cv)

                test_start = int(n_samples*i/n_cv)
                test_end = int(n_samples*(i+1)/n_cv)

                train_indices = list(range(n_samples)[:test_start])\
                                    + list(range(n_samples)[test_end:])

                Xr = X[train_indices, :]
                Yr = Y[train_indices]
                Xt = X[test_start:test_end, :]
                Yt = Y[test_start:test_end]

                self.fit(Xr, Yr, get_history_terms = False)
                Yt_hat = self.predict(Xt, get_history_terms = False)

                Yt_hat = np.squeeze(Yt_hat)
                Y_hat[test_start:test_end] = Yt_hat

                pR2 = self.poisson_pseudoR2(Yt, Yt_hat, np.mean(Yr))
                pR2_cv.append(pR2)

                if verbose > 1:
                    print( 'pR2: ', pR2)
        else:

            cv_kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
            skf  = cv_kf.split(X)


            i=1

            for idx_r, idx_t in skf:
                if verbose > 1:
                    print( '...runnning cv-fold', i, 'of', n_cv)
                i+=1
                Xr = X[idx_r, :]
                Yr = Y[idx_r]
                Xt = X[idx_t, :]
                Yt = Y[idx_t]

                self.fit(Xr, Yr, get_history_terms = False)
                Yt_hat = self.predict(Xt, get_history_terms = False)

                Y_hat[idx_t] = Yt_hat

                pR2 = self.poisson_pseudoR2(Yt, Yt_hat, np.mean(Yr))
                pR2_cv.append(pR2)

                if verbose > 1:
                    print( 'pR2: ', pR2)

        if verbose > 0:
            print("pR2_cv: %0.6f (+/- %0.6f)" % (np.mean(pR2_cv),
                                                 np.std(pR2_cv)/np.sqrt(n_cv)))

        return Y_hat, pR2_cv

    # These two methods implement the above scheme. We don't want to be forced to run the ensemble
    # at the same time as we train the other methods on each fold, so we'll save the 1st-stage predictions for later
    # and use separate methods for training a 1st-stage method and the 2nd-stage method. This will make more sense
    # when we implement this.

    # Basically, the first method is used to train a 1st-stage method, and the 2nd to train a 2nd-stage method.


    def fit_nested_cv(self,X, Y, n_cv_outer=5,n_cv_inner=5, verbose=1, continuous_folds=False):
        """Outputs a list of n_cv_outer prediction vectors Yt_hats, each with length size(Y).

        n_cv_outer is p, in the notation above, and n_cv_inner is k.

        Each prediction vector will be used to train and test a single fold of the ensemble
        in the method `ensemble_cv`. """



        if np.ndim(X)==1:
            X = np.transpose(np.atleast_2d(X))

        if continuous_folds == True:
            raise NotImplementedError()
        else:

            # indices of outer test/train split for each fold
                # It is imperative that the random state be identical to the random state of the Kfold used
                # in ensemble_cv
            cv_kf = KFold(n_splits=n_cv_outer, shuffle=True, random_state=42)
            skf  = cv_kf.split(X)

            i=1
            Y_hat=np.zeros((len(Y),n_cv_outer))
            pR2_cv = list()
            # In outer loop, we rotate the test set through the full dataset
            for idx_r, idx_t in skf:
                if verbose > 1:
                    print( '...runnning outer cv-fold', i, 'of', n_cv_outer)

                Xr_o = X[idx_r, :] # train set input
                Yr_o = Y[idx_r] # train set output
                Xt_o = X[idx_t, :] # test set input
                Yt_o = Y[idx_t] # test set output (used for scoring ensemble only)


                cv_kf_in = KFold(n_splits=n_cv_inner, shuffle=True, random_state=42)
                skf_inner  = cv_kf_in.split(Xr_o)

                j=1
                # In the inner loop, we perform CV to predict the full validation set Yr_o, which will be recorded
                # to be used for ensemble training. THEN we use the full Xr_o to predict values for Xt_o, which will
                # be used for ensemble evaluation.
                for idx_r_inner, idx_t_inner in skf_inner:

                    j+=1
                    Xr = Xr_o[idx_r_inner, :]
                    Yr = Yr_o[idx_r_inner]
                    Xt = Xr_o[idx_t_inner, :]
                    Yt = Yr_o[idx_t_inner]
                    # Predict a fold of the Yr_o (validation)
                    self.fit(Xr, Yr, get_history_terms = False)
                    Yt_hat = self.predict(Xt, get_history_terms = False)

                    full_indices = idx_r[idx_t_inner] # indices of inner loop
                    Y_hat[full_indices,i-1] = Yt_hat

                    Yt_hat.reshape(Yt.shape)
                    pR2 = self.poisson_pseudoR2(Yt, Yt_hat, np.mean(Yr))
                    pR2_cv.append(pR2)

                    if verbose > 1:
                        print( 'pR2: ', pR2)

                # Now predict the ensemble's test set
                self.fit(Xr_o, Yr_o, get_history_terms = False)
                Yt_hat = self.predict(Xt_o, get_history_terms = False)

                Y_hat[idx_t,i-1] = Yt_hat
                pR2 = self.poisson_pseudoR2(Yt_o, Yt_hat, np.mean(Yr_o))
                pR2_cv.append(pR2)

                i+=1

        if verbose > 0:
            print("pR2_cv: %0.6f (+/- %0.6f)" % (np.mean(pR2_cv),
                                                 np.std(pR2_cv)/np.sqrt(n_cv_inner*n_cv_outer)))

        return Y_hat, pR2_cv


    def ensemble_cv(self, X, Y, n_cv_outer=5,n_cv_inner=5,
                    continuous_folds = False, verbose=1, return_first_order_results = False):
        """Outputs the scores and prediction according to a nested cross-validation scheme.



        Input
        =====
        X  = input data
        Y = spiking data
        n_cv = number of cross-validations folds
        continuous_folds = True/False. whether to split folds randomly or to
                        split them in contiguous chunks. The latter is advantageous
                        when using spike history as a covariate to prevent
                        leakage across folds
        return_first_order_results = True/False. Whether to return the results of the first
                                        order methods.

        Returns
        =======
        (Y_hat, pR2_cv); a vector of predictions Y_hat with the
        same dimensions as Y, and a list of pR2 scores on each fold pR2_cv.

        first_order_results: a dictionary of the results of the first-order methods' fits,
        in same format as (Y_hat, pR2_cv).


        """

            # get history terms
        if self.tunemodel == 'lstm':
            X, Y = self.get_all_with_history_keras(X, Y)
        else:
            X, Y = self.get_all_with_history(X, Y)

        first_order_results = dict()

        # first we train all the 1st order methods with the nested_cv scheme
        X_list = list()
        for mod in self.first_order_models:
            if verbose >0:
                print('Running first level model '+str(mod.tunemodel))
            Yt_hat, PR2 = mod.fit_nested_cv(X, Y, continuous_folds = continuous_folds,
                                            n_cv_outer=n_cv_outer, n_cv_inner=n_cv_inner,
                                        verbose = verbose)
            # Put the previous results in a new data matrix
            X_list.append(Yt_hat)

            # save the model fits as attributes
            first_order_results[str(mod.tunemodel)] = dict()
            first_order_results[str(mod.tunemodel)]['Yt_hat'] = Yt_hat
            first_order_results[str(mod.tunemodel)]['PR2'] = PR2

        for x in X_list:
            assert x.shape == (np.size(Y),n_cv_outer)

        if verbose >0:
                print('Running ensemble...')

        if continuous_folds == False:
            # indices of outer test/train split for each fold
            cv_kf = KFold(n_splits=n_cv_outer, shuffle=True, random_state=42)
            skf  = cv_kf.split(X_list[0])

            i=0
            Y_hat=np.zeros(len(Y))
            pR2_cv = list()
            for idx_r, idx_t in skf:

                # Get the first fold from each list
                X = np.array([x[:,i] for x in X_list])
                X = X.transpose()

                Xr = X[idx_r, :]
                Yr = Y[idx_r]
                Xt = X[idx_t, :]
                Yt = Y[idx_t]

                i+=1
                if verbose > 1:
                    print( '...runnning cv-fold', i, 'of', n_cv_outer)

                self.fit(Xr, Yr, get_history_terms = False)
                Yt_hat = self.predict(Xt, get_history_terms = False)
                Y_hat[idx_t] = Yt_hat

                pR2 = self.poisson_pseudoR2(Yt, Yt_hat, np.mean(Yr))
                pR2_cv.append(pR2)

                if verbose > 1:
                    print( 'pR2: ', pR2)
        else:
            raise NotImplementedError()



        if verbose > 0:
            print("pR2_cv: %0.6f (+/- %0.6f)" % (np.mean(pR2_cv),
                                                 np.std(pR2_cv)/np.sqrt(n_cv_outer)))

        if return_first_order_results:
            return Y_hat, pR2_cv, first_order_results
        else:
            return Y_hat, pR2_cv


    def poisson_pseudoR2(self, y, yhat, ynull):
        # This is our scoring function. Implements pseudo-R2
        yhat = yhat.reshape(y.shape)
        eps = np.spacing(1)
        L1 = np.sum(y*np.log(eps+yhat) - yhat)
        L1_v = y*np.log(eps+yhat) - yhat
        L0 = np.sum(y*np.log(eps+ynull) - ynull)
        LS = np.sum(y*np.log(eps+y) - y)
        R2 = 1-(LS-L1)/(LS-L0)
        return R2

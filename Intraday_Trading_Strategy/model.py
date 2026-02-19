from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

"""
Model wrapper for intraday return prediction.

Encapsulates the regression model used to predict
forward returns from engineered features.
"""

class ReturnRegressor:
    """
    The default model is a HistGradientBoostingRegressor
    configured for intraday prediction. 
    
    Optional Ridge and ExtraTreesRegressor models included but commented
    out for comparison purposes in future
    
    """
    def __init__(self, base_model=None):
        if base_model is None:
            # create the instance and assign to self.model
            base_model = HistGradientBoostingRegressor(
            max_iter=500,           
            learning_rate=0.02,     
            max_depth=4,            
            min_samples_leaf=50,    
            l2_regularization=5.0,  
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,    
            max_bins=128           
        )

        # else:    
        self.model = base_model
        
    
    #def __init__(self, base_model=None):
        #if base_model is None:
            #base_model = Ridge(alpha=1.0, random_state=42)
        #self.model = base_model


    # def __init__(self, base_model=None):
    #     if base_model is None:

    #         base_model = ExtraTreesRegressor(
    #             n_estimators=100,
    #             max_depth=5,
    #             max_features='sqrt',
    #             min_samples_split=10,
    #             n_jobs=-1,             # Parallel processing
    #             random_state=42,
    #             verbose=0
    #         )
    #     self.model = base_model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)



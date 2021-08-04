
import pandas as pd

from preprocess import TitanicPreprocess
from config import PathConfig
from dataio import DataIOSteam
from model import TitanicModeling


class TitanicMain(TitanicPreprocess, PathConfig, TitanicModeling, DataIOSteam):
    def __init__(self):
        TitanicPreprocess.__init__(self)
        PathConfig.__init__(self)
        TitanicModeling.__init__(self)
        DataIOSteam.__init__(self)

    def run(self, is_keras=0, n_estimator=100):
        data = self._get_data(self.titanic_path)
        data = self.run_preprocessing(data)
        X, y = self._get_X_y(data)
        if is_keras:
            model, model_info = self.run_keras_modeling(X, y)
            
            return model, model_info
        else:
            model, model_info = self.run_sklearn_modeling(X, y, n_estimator)
            return model, model_info



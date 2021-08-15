
import pandas as pd

class DataIOSteam:

    def _get_data(self, path):
        return pd.read_csv(f'{path}/train.csv')
    
    def _get_X_y(self, data):
        X = data[data.columns[1:]]
        X = X[['Sex', 'Age_band', 'Pclass']]
        y = data['Survived']

        return X, y
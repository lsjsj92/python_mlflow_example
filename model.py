import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from lightgbm import LGBMClassifier

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model


class TitanicModeling:
    def __init__(self):
        pass

    def run_sklearn_modeling(self, X, y, n_estimator):
        model = self._get_rf_model(n_estimator)
        #lgbm_model = self._get_lgbm_model(n_estimator)

        model.fit(X, y)
        #lgbm_model.fit(X, y)

        model_info = {
            'score' : {
                'model_score' :  model.score(X, y)
            },
            'params' : model.get_params()
        }

        return model, model_info

    def run_keras_modeling(self, X, y):
        model = self._get_keras_model()
        model.fit(X, y, epochs=20, batch_size=10)
        #predictions = model.predict(X)
        #print('keras prediction : ', predictions[:5])

        model_info = {
            'score' : {
                'model_score' :  np.float64(  round(model.evaluate(X, y)[1], 2)  )
            },
            'params' : {'epochs':20, 'batch_size':10}
        }

        return model, model_info

    def _get_rf_model(self, n_estimator):
        return RandomForestClassifier(n_estimators=n_estimator, max_depth=5)

    #def _get_lgbm_model(self, n_estimator):
    #    return LGBMClassifier(n_estimators=n_estimator)

    def _get_keras_model(self):
        inp = Input(shape=(3, ), name='inp_layer')
        dense_layer_1 = Dense(32, activation='relu', name="dense_1")
        dense_layer_2 = Dense(16, activation='relu', name="dense_2")
        predict_layer = Dense(1, activation = 'sigmoid', name='predict_layer')

        dense_vector_1 = dense_layer_1(inp)
        dense_vector_2 = dense_layer_2(dense_vector_1)
        predict_vector = predict_layer(dense_vector_2)

        model = Model(inputs=inp, outputs=predict_vector)
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
        return model




        


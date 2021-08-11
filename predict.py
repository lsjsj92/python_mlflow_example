import mlflow
logged_model = 'runs:/33c2494d79e4482ea83811c244596e7f/ml_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = {
    'Col1' : [1,0], 'Col2':[1, 2], 'Col3' : [1, 3]
}

print(loaded_model.predict(pd.DataFrame(data)))
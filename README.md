# python_mlflow example
Python MLflow(management machine learning life-cycle) example tutorial code

### Info & reference
> Blog post (Description)
    - https://lsjsj92.tistory.com/623
> refer : https://github.com/lsjsj92/python_mlflow_example


## Description
- mlflow-env dir
    - MLflow Project & Package example with MLProject, conta.yaml
- main.py
    - Main file on this process
    - start with is_keras argument 
        - 1 : use tensorflow(keras)
        - 0 : use scikit learn
    - Execute titanic modeling
    - Execute MLflow logging
- titanic.py
    - Main of Titniac process
    - data load -> preprocess -> ML/DL modeling -> return model
- model.py
    - Machine Learning or Deep Learning Modeling part
    - Machine Learning : use scikit-learn library or lightgbm
    - Deep Learning : use tensorflow library ( keras )
- preprocess.py
    - Preprocess part
    - Preprocess titanic data
- config.py
    - Config part
- dataio.py
    - Data io part
    - Get data
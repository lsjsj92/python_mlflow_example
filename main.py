import argparse
import sys

import mlflow
from mlflow import sklearn as ml_sklearn
from mlflow import keras as ml_keras
from mlflow import log_artifacts
from mlflow import log_metric, log_metrics
from mlflow import log_param, log_params


from titanic import TitanicMain


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def start(is_keras, n_estimator):
    mlflow.set_experiment('soojin')
    
    titanic = TitanicMain()

    if is_keras:
        #ml_tf.autolog(log_models=True) # 이렇게도 저장 가능
        tf_model, model_info = titanic.run(is_keras)
        log_metrics(model_info['score'])
        log_params(model_info['params'])
        ml_keras.log_model(tf_model, 'tf_keras_model')
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
    else:
        model, model_info = titanic.run(is_keras, n_estimator)
        '''log metric을 하나하나 등록할 때는 아래와 같이 진행
            #log_metric("rf_score", score_info['rf_model_score'])
            #log_metric("lgbm_score", score_info['lgbm_model_score'])
        '''
        # metrics를 한 번에 등록 -> json 형태가 되어야 함
        log_metrics(model_info['score'])
        log_params(model_info['params'])
        ml_sklearn.log_model(model, 'ml_model')
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


if __name__ == "__main__":
    
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        '--is_keras', type=str,
        help="please input 1 or 0"
    )
    argument_parser.add_argument(
        '--n_estimator', type=int, default=100
    )

    args = argument_parser.parse_args()
    try:
        is_keras = _str2bool(args.is_keras)
    except argparse.ArgumentTypeError as E:
        print("ERROR!! please input is_keras 0 or 1")
        sys.exit()

    start(is_keras, args.n_estimator)


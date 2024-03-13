"""
This is a boilerplate pipeline 'rnd_dqn'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import random_collector, dqn_training, online_evaluation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=random_collector,
            inputs=["params:exp_params"],
            outputs="rnd_coll_data", 
            name="random_policy_collector"
        ),

        node(
            func = dqn_training, 
            inputs = ["rnd_coll_data", "params:exp_params"],
            outputs= "rnd_dqn_param", 
            name = "dqn_random_train"
        ),

        node(
            func = online_evaluation, 
            inputs = ["rnd_dqn_param", "params:exp_params"],
            outputs = None, 
            name = "dqn_random_eval"
        )
    ])
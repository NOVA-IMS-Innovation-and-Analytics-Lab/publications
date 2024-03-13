from kedro.pipeline import Pipeline, pipeline, node
from.nodes import tpg_collector
from ..rnd_dqn.nodes import dqn_training, online_evaluation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=tpg_collector,
            inputs=["params:exp_params"],
            outputs="tpg_coll_data", 
            name="tpg_collector"
        ), 

        node(
            func = dqn_training, 
            inputs = ["tpg_coll_data", "params:exp_params"],
            outputs= "tpg_dqn_param", 
            name = "dqn_tpg_train"
        ),

        node(
            func = online_evaluation, 
            inputs = ["tpg_dqn_param","params:exp_params"],
            outputs= None, 
            name = "dqn_tpg_eval"
        ),
    ])

'''
In this assignment we will deal with creating custom model for the class
You have to implement a Simple Policy Gradient Model class (it has to be a subclass of TorchModelV2 and Torch.nn.Module)
Refer https://docs.ray.io/en/latest/rllib-models.html#custom-pytorch-models and for __init__ arguments refer
https://github.com/ray-project/ray/blob/master/rllib/models/torch/torch_modelv2.py,
https://docs.ray.io/en/latest/_modules/ray/rllib/models/modelv2.html. 

Now you have to register this in the ModelCatalog with a name (str)

You also have to implement a loss function similar to the example given and create a policy
using build_torch_policy with this function.

Then you can create a new Trainer by extending PGTrainer class using the with_updates method.

Now you have to run this Trainer using tune.run() with a modified  DEFAULT_CONFIG such that the model dict
contains the name (str) of your model as the value for the 'custom_model' key. 
also make sure you specify the framework as torch and a num of workers > 0
'''

# Your code here
import torch.nn as nn
import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray import tune
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.pg.pg import PGTrainer,DEFAULT_CONFIG
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.policy.sample_batch import SampleBatch

class MyCustomModel(TorchModelV2, nn.Module):
  def __init__(self, *args, **kwargs):
    TorchModelV2.__init__(self, *args, **kwargs)
    nn.Module.__init__(self)
    
  def forward(self, input_dict, state, seq_lens):
    model_out=0
    return model_out,state
  
ModelCatalog.register_custom_model("my_torch_model", MyCustomModel)

def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits)
    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
    return -train_batch[SampleBatch.REWARDS].dot(log_probs)


import torch.optim as optim

MyPolicy = build_torch_policy("MyPolicy",
loss_fn=policy_gradient_loss,
)


MyTrainer = PGTrainer.with_updates(
    default_policy=MyPolicy,
)

ray.init(ignore_reinit_error=True)
tune.run(MyTrainer,config={
    "framework": "torch",
    "model": {
        "custom_model": "my_torch_model",
    },
   "num_workers" : 4,
})

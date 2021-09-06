from ray.rllib.agents.pg.pg import PGTrainer,DEFAULT_CONFIG
'''
The configuration dictionary passed to tune/PGTrainer has lot of parameters that can changed to customize your policy,
the config also has a key named 'model' which describes the neural network to be created 
when it is passed as argument. The 'model' key also uses a dict to describe the network

In this assignment what you have to do is modify/create 'model' dictionary such that the network has
the following specs (activation function is 'relu'). (Refer https://docs.ray.io/en/latest/rllib-models.html#default-model-config-settings
to get an idea of possible modifications.) 

Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
observations (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
fc_1 (Dense)                    (None, 128)          640         observations[0][0]               
__________________________________________________________________________________________________
fc_2 (Dense)                    (None, 256)          33024       fc_1[0][0]                       
__________________________________________________________________________________________________
fc_3 (Dense)                    (None, 512)          131584      fc_2[0][0]                       
__________________________________________________________________________________________________
fc_4 (Dense)                    (None, 64)           32832       fc_3[0][0]                       
__________________________________________________________________________________________________
fc_out (Dense)                  (None, 2)            130         fc_4[0][0]                       
__________________________________________________________________________________________________
value_out (Dense)               (None, 1)            65          fc_4[0][0]                       
==================================================================================================

Note: value_out is used for value based algorithms. As it cannot be removed through config dict,
in cases where we don't use the value_out, we make it share the same layers with the policy so that
a separate network that calculates value (which is not required) is not created.
'''
config = DEFAULT_CONFIG
config['env'] = 'CartPole-v0'
config['framework'] = 'tf2' # using tensorflow to conveniently view model

# modify config
config['model']['fcnet_hiddens']= [128, 256, 512,64]
config["fcnet_activation"]= 'relu'


trainer = PGTrainer(config) # only for seeing the model specs
print(trainer.get_policy().model.base_model.summary())

# Initialize ray and train the policy using tune.
ray.init(ignore_reinit_error=True)
tune.run(PGTrainer,config=congfig)

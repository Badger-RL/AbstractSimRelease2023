# AbstractSim

This repository contains the BadgerRL lab's abstract robosoccer environment simulators, as well as reference policies trained using these simulators. You can configure
the main function of walk_to_goal.py, walk_to_ball.py, and push_ball_to_goal.py to either train a new policy, or load and demonstrate rollouts with an existing policy. 

## Pretrained Policies

Pre-trained Policies are available in the Models directory.


## Tools


### Trying out a policy

You can try out a policy with the PushBallToGoalEnv environment for example with the following command

````
python3 ./envs/push_ball_to_goal.py <path to models folder for policy>

specifically
python3 ./envs/push_ball_to_goal.py /Models/push_ball_to_goal

````

### export_model.py

export_model.py can be used to export a stable-baselines3 saved policy into a format that can be read by the C++ code base. 

When export_model.py is executed, it will create action_policy.h5, vale_policy.h5, shared_policy.h5, and metadata.json. All of these files need to be moved
to the Config directory of the C++ repository to try them out in the c++ environments. 

Note that export model needs both the vector normalize json and the policy file specified. 


### analyze_trajectories.py

This is provided for checking the similarity between the action_means recorded from the C++ policy during rollouts, and the 
action mean predictions made by the stable-baselines3 policy, and the converted keras models. An example trajectory file is provided in  the Examples directory. 

### generate_baselines.py

This allows the baselines to be automatically trained. In the future, the abstract environemnts will be instrumented with psuedorandom seeding, so this process should be possible to exactly reproduce. Right now, there is some variation in the quality of the policies. 

To generate a policy run:
````
python3 generate_baselines.py <name of policy>

for example:
python3 generate_baselines.py push_ball_to_goal
````


## environment
````
python3.9 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt

````

It is recommended to make a pythonvirtual env using requirements.txt before using the code in this repository, this will ensure you can load the saved policies, and ensure the 
policies you train are compatible with other people's environments.  You will need to use python 3.9.

## Running tests

A limited test battery is provided with this project (particularly for logic involving trigonometric calculations) To run the tests, make sure you have the virtual environment set up, then execute the command:

````
python3 -m pytest
````

More tests are always welcome!

# MARL
To start, it is convenient to be in the multi_agent directory so 
````
cd multi_agent
````

From there the file ppo_sb3.py is the file used for training and rendering.
All envs are held in their own folder with the structure ./name_of_env/name_of_env.py
Under this folder is also a place to store policies relating to this env for ease of use.


## To train policies
**Flags**

--train -> use to set in training mode

--env=<NAME_OF_ENV> -> use to set name of ENV to train. NOTE: Use NAME not PATH 

--continuous_actions=<bool> -> Use to set if using continuous actions (default True)
  
--total_timesteps=<x> -> Set number of training timesteps (Default 10 million)
  
--batch_size=<x> -> Set batch size (Default 64)
  
--vec_envs=<x> -> Set number of SB3 vectorized envs (Default 12)

--wandb -> Flag to use weights and biases to track. Must have wandb initialized to use
  
Example of how to train a policy
````
python ppo_sb3.py --train --env=kick_ball_to_goal --total_timesteps=10000 --batch_size=1024 --wandb
````


## To test/visualize
To render policies the env must be specified and the path to the policy to be rendered.
  
For example
````
python ppo_sb3.py --render --env=role_kick --policy_name=role_kick/ppo_sb3_role_kick 
````
  
--render used to set to render mode
  
--env used to choose env to render with
  
--policy_name used to specify the path to the zip file created when training (no .zip at the end)


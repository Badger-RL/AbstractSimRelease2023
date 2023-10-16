import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import VecNormalize

import supersuit as ss
import argparse
import importlib

# from wandb.integration.sb3 import WandbCallback
# import wandb

def parse_args():
    parser = argparse.ArgumentParser()

    # Running options
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--continue_training', action='store_true', default=False)

    parser.add_argument('--prev_policy', type=str, default=None)

    # Env options
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--continuous_actions', type=bool, default=True)

    # Training options
    parser.add_argument('--total_timesteps', type=int, default=10000000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vec_envs', type=int, default=8)

    # Wandb options
    parser.add_argument('--wandb', action='store_true', default=False)
    # Callback options
    parser.add_argument('--annealing', action='store_true', default=False)

    # Condor options
    parser.add_argument('--condor_name', type=str, default=None)

    # Render options
    parser.add_argument('--policy_name', type=str, default=None)

    if not parser.parse_args().env and not parser.parse_args().condor_name:
        raise ValueError("Must specify env")

    return parser.parse_args()

def train(args):
    path = args.env + '.' + args.env
    env = importlib.import_module(path).env

    env = env(env_type='parallel', continuous_actions=args.continuous_actions)
    env = ss.flatten_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=4, base_class='stable_baselines3')
    env = VecMonitor(env)

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10)

    model = PPO(MlpPolicy, env, batch_size=args.batch_size, 
                ent_coef=0.01,
                verbose=1)
    
    model.learn(total_timesteps=args.total_timesteps)


    model.save(args.env)

    print('Done training, If process not finished can exit with Ctrl+C now.')

def render(args):
    # Load model
    if args.policy_name:
        model = PPO.load(args.policy_name)
    else:
        model = PPO.load(args.env)

    path = args.env + '.' + args.env
    env = importlib.import_module(path).env

    # Create environment
    env = env(env_type='parallel', continuous_actions=True)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # Run model
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
        env.render()

def continue_training(args):
    if args.condor_name:
        from env import env
        args.env = args.condor_name
    else:
        path = args.env + '.' + args.env
        env = importlib.import_module(path).env

    run_name = args.env + '_' + time.strftime("%d_%H:%M")

    if args.wandb:
        run = wandb.init(
            project=args.env,
            config={
                "env": args.env,
                "total_timesteps": args.total_timesteps,
                "batch_size": args.batch_size,
            },
            sync_tensorboard=True,
            # monitor_gym=True,
            name=run_name,
        )

    env = env(env_type='parallel', continuous_actions=args.continuous_actions)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, args.vec_envs, base_class='stable_baselines3')
    env = VecMonitor(env)

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10)

    # model = PPO(MlpPolicy, env, batch_size=args.batch_size, 
    #             ent_coef=0.01,
    #             tensorboard_log=f"runs/{run_name}",
    #             verbose=1)
    
    # Load model.zip in current directory
    model = PPO.load(args.prev_policy)
    model.set_env(env)
    
    if args.wandb:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=WandbCallback(
                        verbose=2,
                    )
        )
    else:
        model.learn(total_timesteps=args.total_timesteps)

    model.save("ppo_sb3_" + args.env + "_cont")

    if args.wandb:
        run.finish()

if __name__ == "__main__":
    args = parse_args()

    if args.train:
        train(args)
    if args.render:
        render(args)
    if args.continue_training:
        continue_training(args)
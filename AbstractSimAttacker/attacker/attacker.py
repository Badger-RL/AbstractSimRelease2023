import functools
import time
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import gymnasium as gym
import pygame
import numpy as np
import sys
sys.path.append(sys.path[0] + "/..")

from base import BaseEnv
from supersuit import clip_actions_v0

from base import BaseEnv

def env(render_mode=None, env_type="parallel", continuous_actions=False):
    env = None
    if env_type == "aec":
        env = aec_env(continuous_actions=continuous_actions)
        env = wrappers.OrderEnforcingWrapper(env)
    elif env_type == "parallel":
        env = parallel_env(continuous_actions=continuous_actions)
    else:
        raise ValueError("Invalid env_type: {}".format(env_type))
    return env

def aec_env(render_mode=None, continuous_actions=False):
    env = parallel_env(continuous_actions=continuous_actions, render_mode=render_mode)
    env = parallel_to_aec(env)
    if continuous_actions:
        raise ValueError("Continuous actions not supported for aec_env")
        env = clip_actions_v0(env)
    return env

class parallel_env(BaseEnv):
    metadata = {'render_modes': ['human', 'rgb_array'],
                "render_fps": 30,
                }

    def __init__(self, continuous_actions=False, render_mode='rgb_array'):
        # Init base class
        super().__init__(continuous_actions=continuous_actions)

        '''
        Required:
        - possible_agents
        - action_spaces
        - observation_spaces
        '''
        self.rendering_init = False
        self.render_mode = render_mode

        # agents
        self.possible_agents = ["agent_0"]
        self.agents = self.possible_agents[:]
        self.continous_actions = continuous_actions
        self.teams = [0, 0]
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}

        # action spaces
        if self.continous_actions:
            # 3D vector of (angle, x, y, kick) displacement
            self.action_spaces = {
                agent: gym.spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
                for agent in self.agents
            }
        else:
            # 3D vector of (angle, x, y) displacement
            # 24 possible actions:
            # 3 angle options (-pi/4, 0, p/4), 8 xy pairs
            self.action_spaces = {
                agent: gym.spaces.Discrete(26)
                for agent in self.agents
            }

        # observation spaces
        obs_size = 12
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        self.episode_length = 1500

        self.ball_acceleration = -3
        self.ball_velocity_coef = 4
        self.displacement_coef = 0.06
        self.angle_displacement = 0.25
        self.robot_radius = 20

        self.reward_dict = {
            "goal": 50000, # Team
            "goal_scored": False,
            "ball_to_goal": 10, # Team

            "agent_to_ball": 1, # Individual
            "looking_at_ball": 0.01, # Individual
        }
        
    def get_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    '''
    ego-centric observation:
        origin,
        goal,
        other robots,
        ball
    '''
    def get_obs(self, agent):
        i = self.agent_idx[agent]
        agent_loc = self.robots[i] + [self.angles[i]]

        obs = []

        # Get origin position
        origin_obs = self.get_relative_observation(agent_loc, [0, 0])
        obs.extend(origin_obs)

        # Get goal position
        goal_obs = self.get_relative_observation(agent_loc, [4800, 0])
        obs.extend(goal_obs)

        # Get other positions
        for j in range(len(self.agents)):
            if i == j:
                continue
            robot_obx = self.get_relative_observation(agent_loc, self.robots[j])
            obs.extend(robot_obx)

        # Get ball position
        ball_obs = self.get_relative_observation(agent_loc, self.ball)
        obs.extend(ball_obs)

        return np.array(obs, dtype=np.float32)
    
    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        self.time = 0

        self.ball_velocity = 0
        self.ball_angle = 0

        self.robots = [[np.random.uniform(-4500, 4500), np.random.uniform(-3000, 3000)] for _ in range(len(self.agents))]
        self.angles = [np.random.uniform(-np.pi, np.pi) for _ in range(len(self.agents))]

        self.reward_dict["goal_scored"] = False
        self.previous_distances = [None for _ in range(len(self.agents))]

        self.ball = []
        # self.ball = [np.random.uniform(-4500, 4500), np.random.uniform(-3000, 3000)]
        # Spawn ball around edges of field for more interesting play
        field_length = 4000
        field_height = 2500
        spawn_range = 50

        # Choose a random number 0-3 to determine which edge to spawn the ball on
        edge = np.random.randint(4)
        if edge == 0:
            # Spawn on left edge
            ball_x = np.random.uniform(-field_length, -field_length + spawn_range)
            ball_y = np.random.uniform(-field_height, field_height)
        elif edge == 1:
            # Spawn on top edge
            ball_x = np.random.uniform(-field_length, field_length)
            ball_y = np.random.uniform(field_height - spawn_range, field_height)
        elif edge == 2:
            # Spawn on right edge
            ball_x = np.random.uniform(field_length - spawn_range, field_length)
            ball_y = np.random.uniform(-field_height, field_height)
        else:
            # Spawn on bottom edge
            ball_x = np.random.uniform(-field_length, field_length)
            ball_y = np.random.uniform(-field_height, -field_height + spawn_range)


        self.ball = [ball_x, ball_y]

        # Goal is 4400, [-1000 to 1000]
        # self.ball = [100, 2900]

        observations = {}
        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            observations[agent] = self.get_obs(agent)
        

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        self.time += 1

        previous_locations = {}
        # Save previous locations
        for agent in self.agents:
            i = self.agent_idx[agent]
            previous_locations[agent] = self.robots[i].copy()

        ball_previous_location = self.ball.copy()

        # Update agent locations and ball
        for agent in self.agents:
            actions[agent][3] = 0
            action = actions[agent]
            self.move_agent(agent, action)
            self.update_ball()

        # Calculate rewards
        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            rew[agent] = self.calculate_reward(agent, actions[agent], previous_locations[agent], ball_previous_location)
            terminated[agent] = self.time > self.episode_length
            truncated[agent] = False
            info[agent] = {}

        if self.reward_dict["goal_scored"]:
            # Reset ball
            self.ball = [np.random.uniform(-2500, 2500), np.random.uniform(-1500, 1500)]
            self.reward_dict["goal_scored"] = False

        return obs, rew, terminated, truncated, info

    '''
    Checks if ball is in goal area
    '''
    def goal(self):
        if self.ball[0] > 4400 and self.ball[1] < 500 and self.ball[1] > -500:
            return True
        return False


    def looking_at_ball(self, agent):
        return self.check_facing_ball(agent)
    
    
    def in_opp_goal(self):
        if self.ball[0] < -4400 and self.ball[1] < 1000 and self.ball[1] > -1000:
            return True
        return False

        
    def calculate_reward(self, agent, action, prev_location, prev_ball_location):
        i = self.agent_idx[agent]
        reward = 0

        if self.in_opp_goal():
            return 0

        info_dict = {}
    # testing 
        # Goal - Team
        if self.goal():
            reward += self.reward_dict["goal"]
            self.reward_dict["goal_scored"] = True
            info_dict["goal"] = True
        
        # # Ball to goal - Team
        cur_ball_distance = self.get_distance(self.ball, [4800, 0])
        prev_ball_distance = self.get_distance(prev_ball_location, [4800, 0])
        reward += self.reward_dict["ball_to_goal"] * (prev_ball_distance - cur_ball_distance)
        info_dict["ball_to_goal"] = True

        # reward for stepping towards ball
        cur_distance = self.get_distance(self.robots[i], self.ball)
        prev_distance = self.get_distance(prev_location, self.ball)
        reward += self.reward_dict["agent_to_ball"] * (prev_distance - cur_distance)
        info_dict["agent_to_ball"] = True

        if self.looking_at_ball(agent):
            reward += self.reward_dict["looking_at_ball"]
            info_dict["looking_at_ball"] = True

        return reward
import functools
import math
import time
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import gymnasium as gym
import pygame
import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")
from supersuit import clip_actions_v0

'''
BaseEnv is a base class for all environments. It contains all the necessary
functions for a PettingZoo environment, but does not implement any of the
specifics of the environment. It is meant to be subclassed by other
environments.

Required:
- get_obs(self, agent)
- calculate_reward(self, agent)

Optional:
- reset(self, seed=None, return_info=False, options=None, **kwargs)
- __init__(self, continuous_actions=False, render_mode=None)


'''

class BaseEnv(ParallelEnv):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, continuous_actions=False, render_mode='rgb_array'):
        '''
        Required:
        - possible_agents
        - action_spaces
        - observation_spaces
        '''
        self.rendering_init = False
        self.render_mode = render_mode

        # agents
        self.possible_agents = []
        self.agents = self.possible_agents[:]
        self.continous_actions = continuous_actions

        # action spaces
        if self.continous_actions:
            # 3D vector of (angle, x, y) displacement
            self.action_spaces = {
                agent: gym.spaces.Box(np.array([-1, -1, -1]), np.array([1, 1, 1]), dtype=np.float32)
                for agent in self.agents
            }
        else:
            # 3D vector of (angle, x, y) displacement
            # 26 possible actions:
            # 3 angle options (-pi/4, 0, p/4), 8 xy pairs, 1, stop option, 1 kick option
            self.action_spaces = {
                agent: gym.spaces.Discrete(26)
                for agent in self.agents
            }

        # observation spaces
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Other variables
        self.ball_radius = 10
        self.ball_acceleration = -0.2
        self.ball_velocity_coef = 1
        self.robot_radius = 25

        if self.continous_actions:
            self.displacement_coef = 0.1
            self.angle_displacement = 0.25
        else:
            self.displacement_coef = 10
            self.angle_displacement = 0.05

        self.episode_length = 2000

        self.num_adversaries = 0
        self.kicking_time = 0

        self.kicking_agents = None
        self.opponents = None

        self.opponent_contacted_ball = False
        self.goalie = None


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        pass

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        self.time = 0
        self.adversary_contacted_ball = False

        self.ball_velocity = 0
        self.ball_angle = 0

        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}

        observations = {}
        for agent in self.agents:
            observations[agent] = self.get_obs(agent)
        
        return observations

    def get_obs(self, agent):
        """
        get_obs(agent) returns the observation for agent
        """
        raise NotImplementedError

    def calculate_reward(self, agent):
        """
        calculate_reward(agent) returns the reward for agent
        """
        raise NotImplementedError

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

        for agent in self.agents:
            action = actions[agent]
            self.move_agent(agent, action)
            obs[agent] = self.get_obs(agent)
            rew[agent] = self.calculate_reward(agent)
            terminated[agent] = self.time > self.episode_length
            truncated[agent] = False
            info[agent] = {}

        for adversary in range(self.num_adversaries):
            self.step_adversary(adversary)

        return obs, rew, terminated, truncated, info

    #### STEP UTILS ####
    def move_agent(self, agent, action):
        i = self.agent_idx[agent]

        if self.kicking_time > 0:
            self.kicking_time -= 1

        else:
            if self.continous_actions:
                if action[3] > 0.8:
                    # Kick the ball
                    self.kick_ball(agent)
                    self.kicking_time = 10
                else:
                    old_x = self.robots[i][0]
                    old_y = self.robots[i][1]
                    old_angle = self.angles[i]

                    # scale action for better dynamics
                    action = self.dynamics_action_scale(action)

                    policy_goal_x = self.robots[i][0] + (
                    (
                        (np.cos(self.angles[i]) * np.clip(action[1], -1, 1))
                        + (np.cos(self.angles[i] + np.pi / 2) * np.clip(action[2], -1, 1))
                    )
                    * 200
                    )  # the x component of the location targeted by the high level action
                    policy_goal_y = self.robots[i][1] + (
                        (
                            (np.sin(self.angles[i]) * np.clip(action[1], -1, 1))
                            + (np.sin(self.angles[i] + np.pi / 2) * np.clip(action[2], -1, 1))
                        )
                        * 200
                    )  # the y component of the location targeted by the high level action

                    # Update robot position
                    self.robots[i][0] = (
                        self.robots[i][0] * (1 - self.displacement_coef)
                        + policy_goal_x * self.displacement_coef
                    )  # weighted sums based on displacement coefficient
                    self.robots[i][1] = (
                        self.robots[i][1] * (1 - self.displacement_coef)
                        + policy_goal_y * self.displacement_coef
                    )  # the idea is we move towards the target position and angle

                    # Update robot angle
                    self.angles[i] = self.angles[i] + self.angle_displacement * action[0]

                    # Check for collisions with other robots from current action
                    self.check_collision(agent, old_angle, old_x, old_y)

                    if self.kicking_agents:
                        self.kicking_agents[i] = False
            else:
                # 3 angle options (-pi/4, 0, p/4), 8 xy pairs, 1, stop option, 1 kick option
                angle = (action % 3 - 1) * np.pi / 4
                x = (action // 3) // 3 - 1
                y = (action // 3) % 3 - 1

                # Moving left with no angle change is action = 

                if action == 24 or action == 25:
                    angle = 0
                    x = 0
                    y = 0
                if action == 25:
                    self.kick_ball(agent)


                # Update robot position
                self.angles[i] += angle * self.angle_displacement
                self.robots[i][0] += x * self.displacement_coef
                self.robots[i][1] += y * self.displacement_coef

                # Check for collisions with other robots from current action
                self.check_collision(agent, angle, x, y)

        # Make sure robot is within bounds
        self.robots[i][0] = np.clip(self.robots[i][0], -5200, 5200)
        self.robots[i][1] = np.clip(self.robots[i][1], -3700, 3700)
        
        # Make sure ball is within bounds
        self.ball[0] = np.clip(self.ball[0], -5200, 5200)
        self.ball[1] = np.clip(self.ball[1], -3700, 3700)

    def dynamics_action_scale(self, action):
        # Action is a 4 dimensional vector, (angle, x, y, kick)
        # Unable to move if turning faster than 0.5
        if abs(action[0]) > 0.4:
            action[1] = 0
            action[2] = 0

        # Make moving backwards slower
        if action[1] < 0:
            action[1] *= 0.3

        # Make moving left and right slower
        action[2] *= 0.5

        return action

    def check_collision(self, agent, angle, x, y):
        i = self.agent_idx[agent]
        # Check for collisions
        if self.continous_actions:
            # If continuous actions, angle, x and y are old values
            for j in range(len(self.agents)):
                if i == j:
                    continue

                # Find distance between robots
                robot_location = np.array([self.robots[i][0], self.robots[i][1]])
                other_robot_location = np.array([self.robots[j][0], self.robots[j][1]])
                distance_robots = np.linalg.norm(other_robot_location - robot_location)

                # If collision, move back
                if distance_robots < (self.robot_radius + self.robot_radius) * 7:
                    self.robots[i][0] = x
                    self.robots[i][1] = y
                    self.angles[i] = angle
        
        else:
            for j in range(len(self.agents)):
                if i == j:
                    continue
                # Find distance between robots
                robot_location = np.array([self.robots[i][0], self.robots[i][1]])
                other_robot_location = np.array([self.robots[j][0], self.robots[j][1]])
                distance_robots = np.linalg.norm(other_robot_location - robot_location)

                # If collision, move back
                if distance_robots < (self.robot_radius + self.robot_radius) * 6:
                    self.robots[i][0] -= x * self.displacement_coef
                    self.robots[i][1] -= y * self.displacement_coef
                    self.angles[i] -= angle * self.angle_displacement
                    break

    def update_ball(self):
        # Update ball velocity
        self.ball_velocity += self.ball_acceleration
        self.ball_velocity = np.clip(self.ball_velocity, 0, 100)

        # Update ball position
        self.ball[0] += self.ball_velocity * math.cos(self.ball_angle)
        self.ball[1] += self.ball_velocity * math.sin(self.ball_angle)

        # If ball touches robot, push ball away
        for i in range(len(self.robots)):
            # Check if robot is facing ball
            # if not self.check_facing_ball(self.agents[i]):
            #     # Make it so that ball is not pushed away if robot is not facing ball
            #     continue 

            robot = self.robots[i]
            # Find distance between robot and ball
            robot_location = np.array([robot[0], robot[1]])
            ball_location = np.array([self.ball[0], self.ball[1]])
            distance_robot_ball = np.linalg.norm(ball_location - robot_location)

            # If collision, move ball away
            if distance_robot_ball < (self.robot_radius + self.ball_radius) * 6:
                self.ball_velocity = self.ball_velocity_coef * 10
                self.ball_angle = math.atan2(self.ball[1] - robot[1], self.ball[0] - robot[0])

                # Angle needs to be adapted to be like real robots (do for both sides of 0 degrees)
                # 1) If angle is 15 - 30, change to angle of robot
                # 2) If angle < 15, change to -50
                # 3) If angle > 30, add 50
                # 1:
                if self.ball_angle > np.radians(15) and self.ball_angle < np.radians(30):
                    self.ball_angle = self.angles[i]
                # 2:
                elif self.ball_angle < np.radians(15) and self.ball_angle > np.radians(0):
                    self.ball_angle = np.radians(-50)
                # # 3:
                # elif self.ball_angle > np.radians(30) and self.ball_angle < np.radians(45):
                #     self.ball_angle += np.radians(50)
                # Negative 1:
                elif self.ball_angle < np.radians(-15) and self.ball_angle > np.radians(-30):
                    self.ball_angle = self.angles[i]
                # Negative 2:
                elif self.ball_angle > np.radians(-15) and self.ball_angle < np.radians(0):
                    self.ball_angle = np.radians(50)
                # # Negative 3:
                # elif self.ball_angle < np.radians(-30) and self.ball_angle > np.radians(-45):
                #     self.ball_angle -= np.radians(50)






                self.ball_angle += np.random.normal(0, 1) * np.pi/6
        
        # If ball is in goal, stop ball
        if self.ball[0] > 4400 and (self.ball[1] < 700 and self.ball[1] > -700):
            self.ball_velocity = 0
            # Set ball to center of goal
            self.ball[0] = 4800
            self.ball[1] = 1
        
        # If ball goes out, send to opponent goal
        # if self.ball[0] < -4400 or self.ball[1] > 3000 or self.ball[1] < -3000 or (self.ball[0] > 4400 and (self.ball[1] >= 700 or self.ball[1] <= -700)):
        #     self.ball_velocity = 0
        #     self.ball[0] = -4800
        #     self.ball[1] = 1

    def check_facing_ball(self, agent):
        i = self.agent_idx[agent]
        # Convert from radians to degrees
        robot_angle = math.degrees(self.angles[i]) % 360

        # Find the angle between the robot and the ball
        angle_to_ball = math.degrees(
            math.atan2(self.ball[1] - self.robots[i][1], self.ball[0] - self.robots[i][0])
        )
        # Check if the robot is facing the ball
        req_angle = 10
        angle = (robot_angle - angle_to_ball) % 360

        if angle < req_angle or angle > 360 - req_angle:
            return True
        else:
            return False
        
    '''
    Gets relative position of object to agent
    '''
    def get_relative_observation(self, agent_loc, object_loc):
        # Get relative position of object to agent, returns x, y, angle
        # Agent loc is x, y, angle
        # Object loc is x, y

        # Get relative position of object to agent
        x = object_loc[0] - agent_loc[0]
        y = object_loc[1] - agent_loc[1]
        angle = np.arctan2(y, x) - agent_loc[2]

        # Rotate x, y by -agent angle
        xprime = x * np.cos(-agent_loc[2]) - y * np.sin(-agent_loc[2])
        yprime = x * np.sin(-agent_loc[2]) + y * np.cos(-agent_loc[2])

        return [xprime/10000, yprime/10000, np.sin(angle), np.cos(angle)]
    
    def kick_ball(self, agent):
        if self.check_facing_ball(agent):
            i = self.agent_idx[agent]

            # Find distance between robot and ball
            robot_location = np.array([self.robots[i][0], self.robots[i][1]])
            ball_location = np.array([self.ball[0], self.ball[1]])

            distance_robot_ball = np.linalg.norm(ball_location - robot_location)
            

            # If robot is close enough to ball, kick ball
            if distance_robot_ball < (self.robot_radius + self.ball_radius) * 10:
                self.ball_velocity = 30

                # Find angle between robot and ball
                self.ball_direction = math.degrees(
                    math.atan2(self.ball[1] - self.robots[i][1], self.ball[0] - self.robots[i][0])
                ) % 360

                if self.kicking_agents:
                    self.kicking_agents[i] = True

                

############ RENDERING UTILS ############

    def render_robot(self, agent):
        i = self.agent_idx[agent]
        Field_length = 1200
        render_robot_x = int((self.robots[i][0] / 5200 + 1) * (Field_length / 2))
        render_robot_y = int((self.robots[i][1] / 3700 + 1) * (Field_length / 3))

        # Color = dark red 
        color = (140, 0, 0)

        # Draw robot
        pygame.draw.circle(
            self.field,
            pygame.Color(color[0], color[1], color[2]),
            (render_robot_x, render_robot_y),
            self.robot_radius,
            width=5,
        )

        # Draw robot direction
        pygame.draw.line(
            self.field,
            pygame.Color(50, 50, 50),
            (render_robot_x, render_robot_y),
            (
                render_robot_x + self.robot_radius * np.cos(self.angles[i]),
                render_robot_y + self.robot_radius * np.sin(self.angles[i]),
            ),
            width=5,
        )
        # Add robot number
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(str(i), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (render_robot_x, render_robot_y)
        self.field.blit(text, textRect)

    def render_opponents(self):
        # Pink = 255 51 255
        # Blue = 63 154 246

        color = (63, 154, 246)

        for i in range(len(self.opponents)):
            Field_length = 1200
            render_robot_x = (self.opponents[i][0] / 5200 + 1) * (Field_length / 2)
            render_robot_y = (self.opponents[i][1] / 3700 + 1) * (Field_length / 3)

            pygame.draw.circle(
                self.field,
                pygame.Color(color[0], color[1], color[2]),
                (render_robot_x, render_robot_y),
                self.robot_radius,
                width=5,
            )

    def render_goalie(self):
        # Pink = 255 51 255
        # Blue = 63 154 246

        color = (63, 154, 246)

        Field_length = 1200
        render_robot_x = (self.goalie[0] / 5200 + 1) * (Field_length / 2)
        render_robot_y = (self.goalie[1] / 3700 + 1) * (Field_length / 3)

        pygame.draw.circle(
            self.field,
            pygame.Color(color[0], color[1], color[2]),
            (render_robot_x, render_robot_y),
            self.robot_radius,
            width=5,
        )

    def basic_field(self, _Field_length=1200):
        # you can change : (l_w = line width)
        Field_length = _Field_length
        l_w = 3

        # you can't change (based on the official robocup rule book ratio)
        Field_width = Field_length * (2 / 3)
        Penalty_area_length = Field_length * (1 / 15)
        Penalty_area_width = Field_length * (22 / 90)
        Penalty_cross_distance = Field_length * (13 / 90)
        Center_circle_diameter = Field_length * (15 / 90)
        Penalry_cross_size = Field_length * (1 / 90)
        Border_strip_width = Field_length * (7 / 90)
        goal_post_size = Field_length * (1 / 90)
        goal_area_width = Field_length * (1 / 6)
        goal_area_length = Field_length * (5 / 90)

        Soccer_green = (18, 160, 0)
        self.field.fill(Soccer_green)

        # drawing goal-area(dark green-Left/Right)
        pygame.draw.rect(
            self.field,
            (56, 87, 35),
            [
                Border_strip_width - goal_area_length,
                Field_width / 2 - goal_area_width / 2 - goal_post_size / 2,
                goal_area_length,
                goal_area_width + goal_post_size,
            ],
        )
        pygame.draw.rect(
            self.field,
            (48, 190, 0),
            [
                Field_length - Border_strip_width,
                Field_width / 2 - goal_area_width / 2 - goal_post_size / 2,
                goal_area_length,
                goal_area_width + goal_post_size,
            ],
        )
        # drawing out-line
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Border_strip_width],
            [Field_length - Border_strip_width, Border_strip_width],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Border_strip_width],
            [Border_strip_width, Field_width - Border_strip_width],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Field_width - Border_strip_width],
            [Field_length - Border_strip_width, Field_width - Border_strip_width],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Field_length - Border_strip_width, Field_width - Border_strip_width],
            [Field_length - Border_strip_width, Border_strip_width],
            l_w,
        )
        # drawing center-line
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Field_length / 2, Border_strip_width],
            [Field_length / 2, Field_width - Border_strip_width],
            l_w,
        )
        pygame.draw.circle(
            self.field,
            (255, 255, 255),
            [Field_length / 2, Field_width / 2],
            Center_circle_diameter / 2,
            l_w,
        )
        pygame.draw.circle(
            self.field,
            (255, 255, 255),
            [Field_length / 2, Field_width / 2],
            Penalry_cross_size / 2,
        )
        # drawing keeper_area(left-side)
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Field_width / 2 - Penalty_area_width / 2],
            [
                Border_strip_width + Penalty_area_length,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Field_width / 2 + Penalty_area_width / 2],
            [
                Border_strip_width + Penalty_area_length,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [
                Border_strip_width + Penalty_area_length,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            [
                Border_strip_width + Penalty_area_length,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            l_w,
        )
        # drawing keeper_area(Right-side)
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [
                Field_length - Border_strip_width,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            [
                Field_length - Border_strip_width - Penalty_area_length,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [
                Field_length - Border_strip_width,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            [
                Field_length - Border_strip_width - Penalty_area_length,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [
                Field_length - Border_strip_width - Penalty_area_length,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            [
                Field_length - Border_strip_width - Penalty_area_length,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            l_w,
        )
        # drawing penalty_cross(Left/Right)
        pygame.draw.circle(
            self.field,
            (255, 255, 255),
            [Penalty_cross_distance + Border_strip_width, Field_width / 2],
            Penalry_cross_size / 2,
        )
        pygame.draw.circle(
            self.field,
            (255, 255, 255),
            [
                Field_length - Penalty_cross_distance - Border_strip_width,
                Field_width / 2,
            ],
            Penalry_cross_size / 2,
        )
        # drawing goal-post(grey color-Left/Right)
        pygame.draw.circle(
            self.field,
            (110, 110, 110),
            [
                Border_strip_width,
                Field_width / 2 - goal_area_width / 2 - goal_post_size / 2,
            ],
            goal_post_size / 2,
        )
        pygame.draw.circle(
            self.field,
            (110, 110, 110),
            [
                Border_strip_width,
                Field_width / 2 + goal_area_width / 2 + goal_post_size / 2,
            ],
            goal_post_size / 2,
        )
        pygame.draw.circle(
            self.field,
            (110, 110, 110),
            [
                Field_length - Border_strip_width,
                Field_width / 2 + goal_area_width / 2 + goal_post_size / 2,
            ],
            goal_post_size / 2,
        )
        pygame.draw.circle(
            self.field,
            (110, 110, 110),
            [
                Field_length - Border_strip_width,
                Field_width / 2 - goal_area_width / 2 - goal_post_size / 2,
            ],
            goal_post_size / 2,
        )

    def render(self, mode="human"):
        Field_length = 1200
        time.sleep(0.01)

        if self.rendering_init == False:
            pygame.init()

            self.field = pygame.display.set_mode((Field_length, Field_length * (2 / 3)))

            self.basic_field(Field_length)
            pygame.display.set_caption("Point Targeting Environment")
            self.clock = pygame.time.Clock()

            self.rendering_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.basic_field(Field_length)

        # Render robots
        for agent in self.agents:
            self.render_robot(agent)

        if self.opponents:
            self.render_opponents()

        if self.goalie:
            self.render_goalie()

        # Render ball
        render_ball_x = (self.ball[0] / 5200 + 1) * (Field_length / 2)
        render_ball_y = (self.ball[1] / 3700 + 1) * (Field_length / 3)

        pygame.draw.circle(
            self.field,
            pygame.Color(40, 40, 40),
            (render_ball_x, render_ball_y),
            self.ball_radius,
        )

        pygame.display.update()
        self.clock.tick(60)

#########################################

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Return true if line segments AB and CD intersect, for goal line
    def intersect(A, B, C, D):
        def ccw(A,B,C):
            return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
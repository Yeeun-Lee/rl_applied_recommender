import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Any, List, Sequence, Tuple

"""
** specifications **

data : movielens-small
column names : userId, movieId, rating ...

"""
class RecoEnv(gym.Env):
    metadata = {'render.modes':['human', 'logger']}
    id = 'reco-v0'
    actions = np.eye(5)
    """
    Generate Recommender Enviorment for Reinforcement Learning
    : using gym framework
    """
    def __init__(self,
                 df: pd.DataFrame,
                 item: pd.DataFrame,
                 seed: int = 1):
        self.data = df
        self.item = item

        self.reward = 0.0
        self.done = False
        self.observation = None
        self.action = 0

        # initialize values
        self.user_mean = self.data.groupby('userId').mean().to_dict()['rating']
        self.movie_mean = self.data.groupby('movieId').mean().to_dict()['rating']

        self.max_step = self.data.shape[0] - 2
        self.local_step_number = 0
        self._seed = seed
        self.total_correct_predictions = 0

        # prep for training
        self.data = self.data.values
        self.action_space = spaces.Discrete(len(RecoEnv.actions))
        self.observation_space = spaces.Box(low=-1., high=5.0,
                                            shape=self._get_observation(
                                                step_number=0).shape,
                                            dtype=np.float32
                                            )

    def _get_observation(self, step_number) -> np.ndarray:
        user_id = self.data[step_number, 0]
        movie_id = self.data[step_number, 1]
        user_mean = np.array([self.user_mean.get(user_id, 3.) / 5.], dtype=np.float32)
        movie_mean = np.array([self.movie_mean.get(movie_id, 3.) / 5.], dtype=np.float32)

        return np.concatenate((user_mean, movie_mean))

    def _get_reward(self, action, step_number) -> float:
        users_rating = int(self.data[step_number, 2])
        predicted_rating = int(action) + 1
        prediction_difference = abs(predicted_rating - users_rating)
        reward = 0

        if prediction_difference == 0:
            reward += 1
        else:
            reward += np.log(1. - prediction_difference / 5)  # log loss(neg)
        return reward

    def reset(self) -> np.ndarray:
        self.local_step_number = 0
        self.reward = 0.0
        self.done = False

    def step(self, action: int = 0) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            self.observation = self.reset()
            return self.observation, self.reward, self.done, {}
        self.action = action
        self.reward = self._get_reward(action=action, step_number=self.local_step_number)
        self.observation = self._get_observation(step_number=self.local_step_number)

        if self.reward > 0:
            self.total_correct_predictions +=1
        if self.local_step_number >= self.max_step: # >- ?
            self.done = True
        self.local_step_number +=1
        return self.observation, self.reward, self.done, {}
    def render(self, mode='human')->None:
        if mode=="logger":
            print(f"Env observation at step {self.local_step_number} if \n{self.observation}")

    def close(self)->None:
        self.data = None
        self.item = None
    def seed(self, seed = 1) -> List[int]:
        self._random_state = np.random.RandomState(seed = seed)
        self._seed = seed
        return [seed]

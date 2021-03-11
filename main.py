from train.reco import RecoEnv
from train.model import Model
from train.agents import A2CAgent
import pandas as pd

if __name__=="__main__":
    # read dataset
    ml_movies = pd.read_csv("movielens_small/movies.csv")
    ml_rating = pd.read_csv("movielens_small/ratings.csv")

    # initialize environment with dataset
    env = RecoEnv(ml_rating, ml_movies)

    # model and agent for A2C
    model = Model(num_actions=env.action_space.n)
    agent = A2CAgent(model)
    rewards_sum = agent.test(env)

    rewards_hist, losses = agent.train(env, batch_size=64, updates = 300)

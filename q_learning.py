import numpy as np
import pandas as pd
import random
import lightgbm as lgb
import multiprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
import multiprocessing as mp

# Preprocess Test Data
def preprocess_test_data(test_data):
    test_data['source_h3'] = test_data['source_h3'].astype('category').cat.codes
    test_data['destination_h3'] = test_data['destination_h3'].astype('category').cat.codes
    test_data['part_of_day'] = test_data['part_of_day'].astype('category').cat.codes
    test_data['ride_id'] = test_data.index
    return test_data[['ride_id', 'eta', 'route_distance', 'route_duration', 'source_h3', 'destination_h3', 
                      'month', 'day', 'hour', 'day_of_week', 'part_of_day']]

class PricingEnv:
    def __init__(self, passenger_model, driver_model, test_data):
        self.passenger_model = passenger_model
        self.driver_model = driver_model
        self.test_data = test_data
        self.current_step = 0
        self.state = self._get_state(self.current_step)

    def _get_state(self, step):
        if step >= len(self.test_data):
            return None
        row = self.test_data.iloc[step]
        state = {
            'ride_id': row['ride_id'],
            'eta': row['eta'],
            'route_distance': row['route_distance'],
            'route_duration': row['route_duration'],
            'source_h3': row['source_h3'],
            'destination_h3': row['destination_h3'],
            'month': row['month'],
            'day': row['day'],
            'hour': row['hour'],
            'day_of_week': row['day_of_week'],
            'part_of_day': row['part_of_day'],
        }
        return state

    def reset(self):
        self.current_step = 0
        self.state = self._get_state(self.current_step)
        return self.state

    def step(self, action):
        if self.state is None:
            return None, 0, True, 0
        row = self.test_data.iloc[self.current_step]
        price = action
        
        passenger_features = np.array([[price, row['eta'], row['route_distance'], row['route_duration'],
                                        row['source_h3'], row['destination_h3'], row['month'], row['day'],
                                        row['hour'], row['day_of_week'], row['part_of_day']]])
        passenger_prob = self.passenger_model.predict(passenger_features)[0]

        # Invariant to pickup distance, set value to the avg of the driver acceptance df (1739)
        driver_features = np.array([[price, row['eta'], 1739, row['route_distance'], row['route_duration'],
                                     row['source_h3'], row['destination_h3'], row['month'], row['day'],
                                     row['hour'], row['day_of_week'], row['part_of_day']]])
        driver_prob = self.driver_model.predict(driver_features)[0]

        passenger_accepts = random.random() < passenger_prob
        driver_accepts = random.random() < driver_prob if passenger_accepts else False

        reward = 1 if passenger_accepts and driver_accepts else 0

        self.current_step += 1
        done = self.current_step >= len(self.test_data)
        self.state = self._get_state(self.current_step) if not done else None

        return self.state, reward, done, price, passenger_accepts, driver_accepts

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.9995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def initialize_q_table(self, state_space):
        self.q_table = {}
        for state in state_space:
            state_key = self._state_to_key(state)
            self.q_table[state_key] = np.zeros(len(self.action_space))

    def choose_action(self, state, shared_q_table):
        if state is None:
            return np.random.choice(self.action_space)
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)
        state_key = self._state_to_key(state)
        if state_key not in shared_q_table:
            shared_q_table[state_key] = np.zeros(len(self.action_space))
        return self.action_space[np.argmax(shared_q_table[state_key])]

    def learn(self, state, action, reward, next_state, shared_q_table):
        if state is None or next_state is None:
            return
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        action_index = self.action_space.index(action)

        if next_state_key not in shared_q_table:
            shared_q_table[next_state_key] = np.zeros(len(self.action_space))

        best_next_action = np.argmax(shared_q_table[next_state_key])
        td_target = reward + self.discount_factor * shared_q_table[next_state_key][best_next_action]
        td_error = td_target - shared_q_table[state_key][action_index]
        shared_q_table[state_key][action_index] += self.learning_rate * td_error

        # Log details
        shared_q_table[state_key][action_index] += self.learning_rate * td_error
        print(f"State: {state_key}, Action: {action}, Reward: {reward}, Next State: {next_state_key}, TD Target: {td_target}, TD Error: {td_error}, Q-Value: {shared_q_table[state_key][action_index]}")

        self.exploration_rate *= self.exploration_decay

    def _state_to_key(self, state):
        return tuple(state.values())

def train_episode(env, agent, episode, shared_q_table, lock, metrics):
    state = env.reset()
    total_reward = 0
    done = False
    episode_data = []

    while not done:
        action = agent.choose_action(state, shared_q_table)
        next_state, reward, done, price, passenger_accepts, driver_accepts = env.step(action)
        
        # Acquire the lock before updating the shared Q-table
        with lock:
            agent.learn(state, action, reward, next_state, shared_q_table)
        
        episode_data.append((state['ride_id'], state, reward, price))
        metrics.append([episode, env.current_step, price, reward, passenger_accepts, driver_accepts, agent.exploration_rate])
        state = next_state
        total_reward += reward

    return total_reward, episode_data

def run_parallel_training(env, agent, num_episodes, num_jobs):
    chunk_size = num_episodes // num_jobs
    
    manager = mp.Manager()
    shared_q_table = manager.dict(agent.q_table)
    lock = manager.Lock()
    metrics = manager.list()
    
    rewards_and_data = Parallel(n_jobs=num_jobs)(
        delayed(train_episode_batch)(env, agent, chunk_size, shared_q_table, lock, metrics) for _ in range(num_jobs)
    )
    rewards = [item[0] for sublist in rewards_and_data for item in sublist]
    data = [item[1] for sublist in rewards_and_data for item in sublist]
    return rewards, data, metrics

def train_episode_batch(env, agent, num_episodes, shared_q_table, lock, metrics):
    results = []
    for episode in tqdm(range(num_episodes)):
        results.append(train_episode(env, agent, episode, shared_q_table, lock, metrics))
    return results

def random_policy(env):
    random_rewards = []
    random_states = []
    random_prices = []
    random_metrics = []
    env.reset()
    for step in range(len(env.test_data)):
        state = env._get_state(step)
        if state is None:
            break
        price = random.choice(action_space)
        next_state, reward, done, price, passenger_accepts, driver_accepts = env.step(price)
        random_rewards.append(reward)
        random_states.append(state)
        random_prices.append(price)
        random_metrics.append([step, price, reward, passenger_accepts, driver_accepts])
        if done:
            env.current_step = 0
    return random_rewards, random_states, random_prices, random_metrics

if __name__ == "__main__":
    passenger_model = lgb.Booster(model_file='passenger_model.txt')
    driver_model = lgb.Booster(model_file='driver_model.txt')

    test_quotes_df = pd.read_csv("test.csv")
    
    test_data_preprocessed = preprocess_test_data(test_quotes_df)

    # Assuming passenger_model and driver_model are already defined and trained
    action_space = list(np.arange(1, 100, 10))  # Prices from 1 to 50 in 1 intervals
    agent = QLearningAgent(action_space)
    env = PricingEnv(passenger_model, driver_model, test_data_preprocessed)

    state_space = [env._get_state(step) for step in range(len(env.test_data))]
    agent.initialize_q_table(state_space)
    num_episodes = 1000  # Ensure enough episodes for learning
    num_jobs = multiprocessing.cpu_count()

    # Train in parallel
    rewards, data, q_learning_metrics = run_parallel_training(env, agent, num_episodes, num_jobs)

    print("Training completed.")

    # Save Q-learning metrics to CSV
    q_learning_metrics_df = pd.DataFrame(list(q_learning_metrics), columns=['episode', 'step', 'price', 'reward', 'passenger_accepts', 'driver_accepts', 'exploration_rate'])
    q_learning_metrics_df.to_csv("q_learning_metrics.csv", index=False)

    # Evaluate the Policy
    def evaluate_policy(env, agent, num_episodes=10):
        total_successful_rides = 0
        all_states = []
        all_rewards = []
        all_prices = []
        metrics = []

        for episode in tqdm(range(num_episodes), desc="Evaluating Policy"):
            state = env.reset()
            done = False

            while not done:
                action = agent.choose_action(state, agent.q_table)
                next_state, reward, done, price, passenger_accepts, driver_accepts = env.step(action)
                total_successful_rides += reward
                all_states.append(state)
                all_rewards.append(reward)
                all_prices.append(price)
                metrics.append([episode, env.current_step, price, reward, passenger_accepts, driver_accepts, agent.exploration_rate])
                state = next_state

        avg_successful_rides = total_successful_rides / num_episodes
        print(f"Average Successful Rides: {avg_successful_rides}")
        return all_states, all_rewards, all_prices, metrics

    states, rewards, prices, evaluation_metrics = evaluate_policy(env, agent)

    # Create a DataFrame with states, rewards, and prices
    simulated_data_q_learning = pd.DataFrame(states)
    simulated_data_q_learning['success'] = rewards
    simulated_data_q_learning['price'] = prices

    # Save evaluation metrics to CSV
    evaluation_metrics_df = pd.DataFrame(evaluation_metrics, columns=['episode', 'step', 'price', 'reward', 'passenger_accepts', 'driver_accepts', 'exploration_rate'])
    evaluation_metrics_df.to_csv("q_learning_evaluation_metrics.csv", index=False)

    random_rewards, random_states, random_prices, random_metrics = random_policy(env)

    # Create a DataFrame with random policy states, rewards, and prices
    simulated_data_random_policy = pd.DataFrame(random_states)
    simulated_data_random_policy['success'] = random_rewards
    simulated_data_random_policy['price'] = random_prices

    # Save random policy metrics to CSV
    random_metrics_df = pd.DataFrame(random_metrics, columns=['step', 'price', 'reward', 'passenger_accepts', 'driver_accepts'])
    random_metrics_df.to_csv("random_policy_metrics.csv", index=False)

    # Save results
    simulated_data_random_policy.to_csv("simulated_data_random_policy.csv", index=False)
    simulated_data_q_learning.to_csv("simulated_data_q_learning.csv", index=False)
    
    print("Simulation completed and results saved.")

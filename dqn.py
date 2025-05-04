import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import pygame
from GameEnv import GameEnv  # Import your game environment
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_freq = 100
        
        # Memory for experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Tracking metrics
        self.train_episodes = 0
        self.training_history = {'episode': [], 'reward': [], 'epsilon': []}
    
    def _build_model(self):
        """Neural Network for Deep-Q learning Model."""
        model = tf.keras.Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Training with experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        targets = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        target_next = self.target_model.predict(next_states, verbose=0)
        
        # Update target Q-values
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['episode'], self.training_history['reward'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['episode'], self.training_history['epsilon'])
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def train_agent(env, agent, episodes=1000, max_steps=1000, show_every=10, save_freq=10):
    """Train the agent with responsive Pygame window"""
    scores = []
    running = True
    episode = 0
    
    try:
        while running and episode < episodes:
            episode += 1
            state, _ = env.reset()
            total_reward = 0
            step = 0
            done = False
            
            # Process Pygame events at the start of each episode
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
                
            # Show progress message
            if episode % show_every == 0:
                print(f"Starting episode {episode}/{episodes}, ε={agent.epsilon:.4f}")
            
            # Episode loop
            while not done and step < max_steps:
                # Process Pygame events during each step
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                
                if not running:
                    break
                
                # Agent selects and performs action
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience and update state/reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Always render to see the game
                env.render()
                pygame.display.update()
                
                # Train the agent
                agent.replay()
                
                # Update target network periodically
                if agent.train_episodes % agent.update_target_freq == 0:
                    agent.update_target_model()
                
                # Control frame rate for visualization
                pygame.time.delay(10)  # Small delay for visualization
                step += 1
            
            # End of episode
            agent.train_episodes += 1
            scores.append(total_reward)
            
            # Update training history
            agent.training_history['episode'].append(episode)
            agent.training_history['reward'].append(total_reward)
            agent.training_history['epsilon'].append(agent.epsilon)
            
            # Print episode results
            print(f"Episode: {episode}/{episodes}, Score: {total_reward:.2f}, Steps: {step}")
            
            # Save model periodically
            if episode % save_freq == 0:
                agent.save(f"car_dqn_model_{episode}.weights.h5")
                agent.plot_training_history()
        
        # Final save
        agent.save("car_dqn_model_final.weights.h5")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    return scores

def test_agent(env, agent, episodes=10, max_steps=1000):
    """Test the trained agent"""
    running = True
    episode = 0
    
    try:
        while running and episode < episodes:
            episode += 1
            state, _ = env.reset()
            total_reward = 0
            step = 0
            done = False
            
            print(f"Testing episode {episode}/{episodes}")
            
            while not done and step < max_steps:
                # Process Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                
                if not running:
                    break
                
                # Choose action without exploration
                action = agent.act(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
                
                # Render the game
                env.render()
                pygame.display.update()
                pygame.time.delay(30)  # Slow down for better visualization
                
                step += 1
                
            print(f"Test Episode: {episode}/{episodes}, Score: {total_reward:.2f}, Steps: {step}")
            
    except KeyboardInterrupt:
        print("Testing interrupted by user")

if __name__ == "__main__":
    # Create environment
    env = GameEnv()
    
    # Get dimensions from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    # agent.load("car_dqn_model_final.weights.h5")


    start = time.time()
    print("START:",start)
    # Train agent
    print("Starting testing... (Press Ctrl+C to stop)")
    scores = train_agent(env, agent, episodes=50)

    stop = time.time()
    print("END:",stop)
    print("Total Train Time: ", (stop-start))
    

    # agent.load("car_dqn_model_final.weights.h5")
    # # Plot final training history
    # agent.plot_training_history()

    # start = time.time()
    # print("START:",start)
    
    # # Test trained agent
    # print("Starting testing... (Press Ctrl+C to stop)")
    # test_agent(env, agent, episodes=20)

    # stop = time.time()
    # print("END:",stop)
    # print("Total Test Time: ", (stop-start))
    
    # Close environment
    env.close()
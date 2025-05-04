import os
import pickle
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.encoding import bernoulli
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax
from GameEnv import GameEnv

def build_network():
    """Build a network with the same architecture as the trained one."""
    network = Network(dt=1.0)

    # Layers of neurons.
    inpt = Input(n=5, traces=True)  # Using 5 to match observation dimensions
    middle = LIFNodes(n=100, traces=True)
    out = LIFNodes(n=3, refrac=0, traces=True)

    # Connections between layers.
    inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
    middle_out = Connection(
        source=middle,
        target=out,
        wmin=0,
        wmax=1,
        update_rule=None,  # No learning rule needed for testing
        nu=1e-1,
        norm=0.5 * middle.n,
    )

    # Add all layers and connections to the network.
    network.add_layer(inpt, name="Input Layer")
    network.add_layer(middle, name="Hidden Layer")
    network.add_layer(out, name="Output Layer")
    network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
    network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")
    
    return network

def load_checkpoint(filepath):
    """Load a network checkpoint from file."""
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file {filepath} not found.")
        return None, None, None
    
    try:
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create a new network
        network = build_network()
        
        # Load the saved state
        network.load_state_dict(checkpoint['network_state'])
        
        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        if 'training_time' in checkpoint:
            print(f"Model was trained for: {format_time(checkpoint['training_time'])}")
        
        return network, checkpoint['episode'], checkpoint.get('rewards', [])
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None

def format_time(seconds):
    """Format time in seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def run_test(network, test_episodes=10, render=True, delay=30, save_video=False):
    """Run test episodes with the loaded network."""
    if network is None:
        print("No network loaded. Cannot run tests.")
        return []
    
    # Initialize environment
    environment = GameEnv()
    environment.reset()
    
    # Create pipeline
    pipeline = EnvironmentPipeline(
        network,
        environment,
        encoding=bernoulli,
        action_function=select_softmax,
        output="Output Layer",
        time=10,
        history_length=1,
        delta=1,
        plot_interval=None,
        render_interval=None,  # We'll handle rendering manually
    )
    
    # Make sure learning is turned off
    pipeline.network.learning = False
    
    # Video recording setup
    if save_video:
        import cv2
        video_writer = None
        frame_size = (800, 600)  # Match your game window size
        video_out = cv2.VideoWriter(
            'test_playback.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30, 
            frame_size
        )
    
    # Run tests
    test_rewards = []
    collision_count = 0
    lane_deviation_count = 0
    step_counts = []
    
    print(f"\n=== RUNNING {test_episodes} TEST EPISODES ===")
    
    for episode in range(test_episodes):
        episode_start = time.time()
        pipeline.reset_state_variables()
        observation, info = environment.reset()
        
        total_reward = 0
        is_done = False
        step_count = 0
        
        while not is_done:
            # Get action from the network
            result = pipeline.env_step()
            
            # Render if requested
            if render:
                environment.render()
                if save_video:
                    # Convert pygame surface to numpy array for opencv
                    screen = pygame.display.get_surface()
                    frame = pygame.surfarray.array3d(screen)
                    frame = np.transpose(frame, (1, 0, 2))  # Transpose to match cv2 format
                    video_out.write(frame)
            
            # Process step
            pipeline.step(result)
            
            reward = result[1]
            total_reward += reward
            is_done = result[2]
            step_count += 1
            
            # Check for collision or lane deviation
            if reward <= -10:  # Collision reward
                collision_count += 1
            elif reward < 0:  # Lane deviation penalty
                lane_deviation_count += 1
            
            # Add delay for visualization
            if render:
                pygame.time.wait(delay)
            
            # Handle window events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_done = True
        
        # Record episode metrics
        test_rewards.append(total_reward)
        step_counts.append(step_count)
        episode_time = time.time() - episode_start
        
        print(f"Episode {episode}: Reward={total_reward:.2f}, Steps={step_count}, Time={format_time(episode_time)}")
    
    # Clean up video writer if used
    if save_video and video_out is not None:
        video_out.release()
        print(f"Video saved to test_playback.mp4")
    
    # Calculate and display statistics
    print("\n=== TEST RESULTS ===")
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Max Reward: {np.max(test_rewards):.2f}")
    print(f"Min Reward: {np.min(test_rewards):.2f}")
    print(f"Standard Deviation: {np.std(test_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(step_counts):.2f} steps")
    print(f"Collisions: {collision_count}")
    print(f"Lane Deviations: {lane_deviation_count}")
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(test_rewards, 'g-o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Test Episode Rewards')
    plt.grid(True)
    plt.savefig('test_results.png')
    plt.close()
    
    return test_rewards

def list_checkpoints(directory="checkpoints"):
    """List all available checkpoint files."""
    if not os.path.exists(directory):
        print(f"Checkpoint directory {directory} not found.")
        return []
    
    checkpoints = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    if not checkpoints:
        print("No checkpoint files found.")
        return []
    
    print("\n=== AVAILABLE CHECKPOINTS ===")
    for i, checkpoint in enumerate(checkpoints):
        print(f"{i+1}. {checkpoint}")
    
    return checkpoints

def main():
    """Main function to run the test program."""
    print("SNN Car Game Checkpoint Testing Program")
    print("======================================")
    
    # List available checkpoints
    checkpoints = list_checkpoints()
    if not checkpoints:
        print("No checkpoints found to test. Exiting.")
        return
    
    # Let user select a checkpoint
    choice = input("\nEnter checkpoint number to load (or press Enter for final_trained_model if available): ")
    
    if choice.strip() == "":
        # Default to final trained model
        checkpoint_path = "checkpoints/final_trained_model.pkl"
        if not os.path.exists(checkpoint_path):
            print("final_trained_model.pkl not found. Please select a checkpoint number.")
            return
    else:
        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(checkpoints):
                print("Invalid selection. Exiting.")
                return
            checkpoint_path = os.path.join("checkpoints", checkpoints[choice_idx])
        except ValueError:
            print("Invalid input. Exiting.")
            return
    
    # Load the selected checkpoint
    network, episode, rewards = load_checkpoint(checkpoint_path)
    if network is None:
        return
    
    # Test configuration
    test_episodes = input("Enter number of test episodes (default: 10): ").strip()
    test_episodes = int(test_episodes) if test_episodes.isdigit() else 10
    
    render = input("Render gameplay? (y/n, default: y): ").lower().strip() != 'n'
    
    save_video = input("Save gameplay video? (y/n, default: n): ").lower().strip() == 'y'
    if save_video:
        try:
            import cv2
        except ImportError:
            print("OpenCV (cv2) is required for video recording but not installed.")
            print("Install it with: pip install opencv-python")
            save_video = False
    
    speed = input("Game speed (1=slow, 2=normal, 3=fast, default: 2): ").strip()
    if speed == '1':
        delay = 50  # Slow
    elif speed == '3':
        delay = 10  # Fast
    else:
        delay = 30  # Normal
    
    # Run tests
    test_rewards = run_test(network, test_episodes, render, delay, save_video)
    
    # Save test results
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pkl', '')
    results = {
        'checkpoint': checkpoint_path,
        'test_episodes': test_episodes,
        'rewards': test_rewards,
        'mean_reward': np.mean(test_rewards),
        'std_reward': np.std(test_rewards),
    }
    
    with open(f"test_results_{checkpoint_name}.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nTest results saved to test_results_{checkpoint_name}.pkl")
    
    # Cleanup pygame
    pygame.quit()

if __name__ == "__main__":
    # Initialize pygame if not already initialized
    if not pygame.get_init():
        pygame.init()
    main()
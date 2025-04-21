import pygame
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

pygame.init()

width = 800
height = 600
screen = pygame.display.set_mode((width, height))
car_width = 56

car_img = pygame.image.load('car2.jpg')
grass = pygame.image.load('grass.jpg')
yellow_strip = pygame.image.load('yellow_strip.jpg')
strip = pygame.image.load('strip.jpg')

bg_1 = pygame.image.load('bg_1.jpg')
bg_2 = pygame.image.load('bg_2.jpg')
bg_3 = pygame.image.load('bg_3.jpg')
bg_1 = pygame.transform.scale(bg_1, (width, height))
bg_2 = pygame.transform.scale(bg_2, (width, height))
bg_3 = pygame.transform.scale(bg_3, (width, height))

clock = pygame.time.Clock()

pygame.display.set_caption("2D_CAR")


def get_random_x():
    ranges = [(200, 240), (335, 375), (525, 630)]
    chosen_range = random.choice(ranges)
    return random.randrange(chosen_range[0], chosen_range[1] + 1)


def bg_img(bg_state):
    if bg_state == 1:
        screen.blit(bg_2, (0, 0))
    elif bg_state == 2:
        screen.blit(bg_3, (0, 0))
    else:
        screen.blit(bg_1, (0, 0))


def car(x, y):
    screen.blit(car_img, (x, y))


def obstacle(obs_x, obs_y, obs):
    if obs == 0:
        obs_pic = pygame.image.load("car3.jpg")
    elif obs == 1:
        obs_pic = pygame.image.load("car5.jpg")

    screen.blit(obs_pic, (obs_x, obs_y))


class GameEnv(gym.Env):
    def __init__(self):
        super(GameEnv, self).__init__()

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(low=np.array([0, 0, -750, 0, 1]),
                                            high=np.array([width, width, height, 1, 3]),
                                            dtype=np.float32)

        self.width = width
        self.height = height
        self.car_width = car_width
        self.car_img = pygame.transform.scale(pygame.image.load('car2.jpg'), (car_width, int(car_width * 2.125)))
        self.obstacle_images = [
            pygame.transform.scale(pygame.image.load("car3.jpg"), (car_width, int(car_width * 2.125))),
            pygame.transform.scale(pygame.image.load("car5.jpg"), (car_width, int(car_width * 2.125)))
        ]
        self.grass = pygame.transform.scale(pygame.image.load('grass.jpg'), (width, height))
        self.yellow_strip = pygame.image.load('yellow_strip.jpg')
        self.strip = pygame.image.load('strip.jpg')
        self.bg_images = [
            pygame.transform.scale(pygame.image.load('bg_1.jpg'), (width, height)),
            pygame.transform.scale(pygame.image.load('bg_2.jpg'), (width, height)),
            pygame.transform.scale(pygame.image.load('bg_3.jpg'), (width, height))
        ]

        self.clock = pygame.time.Clock()
        self.reset()

    def _get_obs(self):
        return np.array([self.x, self.obs_x, self.obs_y, self.enemy, self.bg_state], dtype=np.float32)

    def _get_reward(self):
        reward = 0.1
        if self.bumped:
            reward = -10
        elif self.obs_y > self.height:
            reward = 5
        return reward

    def step(self, action):
        self.x_change = 0
        if action == 0:
            self.x_change = -5
        elif action == 1:
            self.x_change = 5

        self.x += self.x_change

        if self.x > (675 - self.car_width + 5) or self.x < (120 - 5):
            self.bumped = True

        self.obs_y += self.obstacle_speed
        
        if self.obs_y > self.height:
            self.obs_y = 0 - self.enemy_height
            self.obs_x = get_random_x()
            self.enemy = random.randrange(0, 2)

        car_rect = self.car_img.get_rect(topleft=(self.x, self.y))
        obstacle_rect = self.obstacle_images[self.enemy].get_rect(topleft=(self.obs_x, self.obs_y))

        if car_rect.colliderect(obstacle_rect):
            self.bumped = True

        self.frame_counter += 1
        if self.frame_counter >= self.change_interval:
            self.bg_state += 1
            if self.bg_state > 3:
                self.bg_state = 1
            self.frame_counter = 0

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self.bumped
        truncated = False
        info = {}

        self._render_frame()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bumped = False
        self.x_change = 0
        self.y_change = 0
        self.obstacle_speed = 10
        self.enemy = random.randrange(0, 2)
        self.obs_x = get_random_x()
        self.obs_y = -750
        self.enemy_width = self.car_width
        self.enemy_height = int(self.car_width * 2.125)
        self.x = 370
        self.y = 400
        self.bg_state = 1
        self.frame_counter = 0
        self.change_interval = 13
        observation = self._get_obs()
        info = {}
        return observation, info

    def _render_frame(self):
        screen.fill((119, 119, 119))
        self._draw_background(self.bg_state)
        screen.blit(self.car_img, (self.x, self.y))
        screen.blit(self.obstacle_images[self.enemy], (self.obs_x, self.obs_y))
        pygame.display.flip()

    def _draw_background(self, bg_state):
        if bg_state == 1:
            screen.blit(self.bg_images[1], (0, 0))
        elif bg_state == 2:
            screen.blit(self.bg_images[2], (0, 0))
        else:
            screen.blit(self.bg_images[0], (0, 0))

    def render(self):
        self._render_frame()

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    observation, info = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.action_space.sample()
        new_observation, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {new_observation}, Reward: {reward}, Terminated: {terminated}")
        env.render()
        clock.tick(30)

    env.close()
import pygame
import random
pygame.init()


width = 800
height = 600
screen  = pygame.display.set_mode((width,height))
car_width = 56

#loading the image
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

# time module
clock = pygame.time.Clock()


#for the caption
pygame.display.set_caption("2D_CAR")


def get_random_x():
    ranges = [(200, 240), (335, 375), (525, 630)]
    chosen_range = random.choice(ranges)
    return random.randrange(chosen_range[0], chosen_range[1] + 1)


#background img
def bg_img(bg_state):
    # screen.blit(grass,(0,0))
    # screen.blit(grass,(700,0))

    # screen.blit(strip,(120,0))
    # screen.blit(strip,(675,0))

    # mid1 = 476
    # mid2 = 278

    # screen.blit(yellow_strip,(mid1,-50))
    # screen.blit(yellow_strip,(mid1,50))
    # screen.blit(yellow_strip,(mid1,150))
    # screen.blit(yellow_strip,(mid1,250))
    # screen.blit(yellow_strip,(mid1,350))
    # screen.blit(yellow_strip,(mid1,450))
    # screen.blit(yellow_strip,(mid1,550))


    # screen.blit(yellow_strip,(mid2,-50))
    # screen.blit(yellow_strip,(mid2,50))
    # screen.blit(yellow_strip,(mid2,150))
    # screen.blit(yellow_strip,(mid2,250))
    # screen.blit(yellow_strip,(mid2,350))
    # screen.blit(yellow_strip,(mid2,450))
    # screen.blit(yellow_strip,(mid2,550))
    if bg_state == 1:
        screen.blit(bg_2,(0,0))
    elif bg_state == 2:
        screen.blit(bg_3,(0,0))
    else:
        screen.blit(bg_1,(0,0))


# image_appearing
def car(x,y):
    screen.blit(car_img,(x,y))    #what image should appear on the screen

def obstacle(obs_x, obs_y, obs):
    if obs == 0:
        obs_pic = pygame.image.load("car3.jpg")
    elif obs == 1:
        obs_pic = pygame.image.load("car5.jpg")

    screen.blit(obs_pic, (obs_x,obs_y))

#game loop 
def game_loop(): 
    bumped = False
    x_change = 0

    bg_state = 1
    frame_counter = 0
    change_interval = 13 # change every 30 frames


    y_change = 0
    obstacle_speed = 10
    enemy = random.randrange(0,2)
    obs_x = get_random_x()
    obs_y = -750
    enemy_width = 56
    enemy_height = 125

    x = 370
    y = 400


    while not bumped: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                bumped = True

        # moving in x-y coordinates
        if event.type == pygame.KEYDOWN: 
            if event.key == pygame.K_LEFT:
                x_change = -5
            if event.key == pygame.K_RIGHT:
                x_change = 5
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                x_change = 0

        x += x_change

        #background color
        screen.fill((119,119,119))

        bg_img(bg_state)

        

        

        obs_y -= obstacle_speed/4
        obstacle(obs_x,obs_y, enemy)
        obs_y += obstacle_speed


        # calling car function
        car(x,y)

        print(obs_x)

        if x > (675- car_width) or x < 120:
            bumped = True

        if obs_y > height:
            obs_y = 0 - enemy_height
            obs_x = get_random_x()


        pygame.display.update()
        clock.tick(100)

        frame_counter += 1
        if frame_counter >= change_interval:
            bg_state += 1
            if bg_state > 3:
                bg_state = 1
            frame_counter = 0


if __name__ == '__main__':
    game_loop()
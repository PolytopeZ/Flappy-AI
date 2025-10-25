import pygame
import random
import sys
import time
import os

# ******** Constants ********
FPS = 60
WIN_WIDTH = 600
WIN_HEIGHT = 800

GRAVITY = 0.25
JUMP_STRENGTH = -6

PIPE_WIDTH = 50
PIPE_GAP = 150
PIP_SPEED = 3

COLOR_BG = (32, 32, 32)
COLOR_PIPE = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)


# ******** Classes ********
class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tick_count = 0
        self.vel = 0
        self.radius = 15
        self.alive = True

    def jump(self):
        self.vel = JUMP_STRENGTH

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel

        # Check for out of bounds
        if self.y > WIN_HEIGHT or self.y < 0:
            self.alive = False

    def draw(self, win):
        pygame.draw.circle(win, (255, 255, 0), (self.x, self.y), self.radius)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap = PIPE_GAP
        self.y = random.randint(150,  WIN_HEIGHT - 150)

    def update(self):
        self.x -= PIP_SPEED

    def draw(self, win):
        pygame.draw.rect(win, COLOR_PIPE, (self.x, 0,
                         self.width, self.y - self.gap // 2))
        pygame.draw.rect(win, COLOR_PIPE, (self.x, self.y + self.gap //
                         2, self.width, WIN_HEIGHT - (self.y + self.gap // 2)))

    def collide_with(self, bird):
        if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + self.width:
            if bird.y - bird.radius < self.y - self.gap // 2 or bird.y + bird.radius > self.y + self.gap // 2:
                return True
        return False


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Fally UwU")
        self.clock = pygame.time.Clock()
        self.bird = Bird(100, WIN_HEIGHT // 2)
        self.pipes = [Pipe(WIN_WIDTH + i * 300) for i in range(3)]
        self.score = 0
        self.font = pygame.font.SysFont("jetbrainsmononlnfpmedium", 24)

    def update(self):
        # Updates entities
        self.bird.update()
        for pipe in self.pipes:
            pipe.update()

        # Add pipes
        if self.pipes[0].x + PIPE_WIDTH < 0:
            self.pipes.pop(0)
            self.pipes.append(Pipe(WIN_WIDTH + 200))
            self.score += 1

        # Check for collisions
        for pipe in self.pipes:
            if pipe.collide_with(self.bird):
                self.bird.alive = False

    def draw(self):
        self.screen.fill(COLOR_BG)

        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.bird.draw(self.screen)

        text = self.font.render(f"Score: {self.score}", True, COLOR_WHITE)
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def reset(self):
        self.bird = Bird(100, WIN_HEIGHT // 2)
        self.pipes = [Pipe(WIN_WIDTH + i * 300) for i in range(3)]
        self.score = 0

    def run(self):
        running = True

        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if not self.bird.alive:
                        self.reset()
                    else:
                        self.bird.jump()

            if self.bird.alive:
                self.update()
            else:
                self.reset()
            self.draw()


if __name__ == "__main__":
    Game().run()

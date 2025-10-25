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

BIRD_COUNT = 10


# ******** Utils ********
def generate_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


# ******** Classes ********
class Bird:
    def __init__(self, x, y, color=(255, 255, 0)):
        self.x = x
        self.y = y
        self.tick_count = 0
        self.vel = 0
        self.radius = 15
        self.alive = True
        self.color = color
        self.score = 0

    def jump(self):
        self.vel = JUMP_STRENGTH

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel

        # Check for out of bounds
        if self.y > WIN_HEIGHT or self.y < 0:
            self.alive = False

    def think(self, pipe):
        # Simple heuristic for now
        if pipe.x + pipe.width > self.x:
            if self.y > pipe.y + 10:
                self.jump()

    def draw(self, win):
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)


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
        self.birds = [Bird(100, WIN_HEIGHT // 2, generate_random_color())
                      for _ in range(BIRD_COUNT)]
        self.pipes = [Pipe(WIN_WIDTH + i * 300) for i in range(3)]
        self.score = 0
        self.generation = 1
        self.font = pygame.font.SysFont("jetbrainsmononlnfpmedium", 24)

    def update(self):
        # Searching for the next pipe
        next_pipe = None
        for pipe in self.pipes:
            # since birds spawn at the same x
            if pipe.x + pipe.width > self.birds[0].x:
                next_pipe = pipe
                break

        # Updates entities
        # First update birds
        alive_count = 0
        for bird in self.birds:
            if bird.alive:
                bird.think(next_pipe)  # Todo change this
                bird.update()
                # Check if collide
                for pipe in self.pipes:
                    if pipe.collide_with(bird):
                        bird.alive = False
                if bird.alive:
                    alive_count += 1

        # Second update pipes
        for pipe in self.pipes:
            pipe.update()

        # Add pipes
        if self.pipes[0].x + PIPE_WIDTH < 0:
            self.pipes.pop(0)
            self.pipes.append(Pipe(WIN_WIDTH + 200))
            self.score += 1
            for bird in self.birds:
                bird.score = self.score

        return alive_count

    def draw(self):
        self.screen.fill(COLOR_BG)
        for pipe in self.pipes:
            pipe.draw(self.screen)
        for bird in self.birds:
            if bird.alive:
                bird.draw(self.screen)

        text = self.font.render(
            f"Score: {self.score}, Gen: {self.generation}", True, COLOR_WHITE)
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def reset(self):
        self.birds = [Bird(100, WIN_HEIGHT // 2, generate_random_color())
                      for _ in range(BIRD_COUNT)]
        self.pipes = [Pipe(WIN_WIDTH + i * 300) for i in range(3)]
        self.score = 0
        self.generation += 1

    def run(self):
        running = True

        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            alive_count = self.update()

            if alive_count == 0:
                self.reset()

            self.draw()


if __name__ == "__main__":
    Game().run()

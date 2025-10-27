import pygame
import random
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# ******** Constants ********
FPS = 60
WIN_WIDTH = 600
WIN_HEIGHT = 800

GRAVITY = 0.25
JUMP_STRENGTH = -6
JUMP_COOLDOWN = 15

PIPE_WIDTH = 50
PIPE_GAP = 150
PIP_SPEED = 3

COLOR_BG = (32, 32, 32)
COLOR_PIPE = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ELITE = (0, 102, 204)
COLOR_GRAY = (160, 160, 160)

BIRD_COUNT = 50
MUTATION_RATE = 0.08
CROSSOVER_RATE = 0.7

SAVE_FILE = "best_bird.npz"


# ******** Neural Network ********
class NeuralNetwork:
    def __init__(self, input_size=5, hidden_size=6, output_size=1):
        self.w1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.random.randn(hidden_size, 1)
        self.w2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.random.randn(output_size, 1)

    def predict(self, inputs):
        # z1 = w1 * x + b1
        # a1 = tanh(z1)
        # z2 = w2 * a1 + b2
        # output = sigmoid(z2)
        x = np.array(inputs).reshape(-1, 1)
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        output = 1 / (1 + np.exp(-z2))  # Sigmoid
        return output[0, 0]

    def copy(self):
        nn = NeuralNetwork()
        nn.w1 = np.copy(self.w1)
        nn.b1 = np.copy(self.b1)
        nn.w2 = np.copy(self.w2)
        nn.b2 = np.copy(self.b2)
        return nn

    def mutate(self, rate=MUTATION_RATE):
        def mutate_matrix(mat):
            mutation_mask = np.random.rand(*mat.shape) < rate
            mat += mutation_mask * np.random.randn(*mat.shape) * 0.5

        mutate_matrix(self.w1)
        mutate_matrix(self.b1)
        mutate_matrix(self.w2)
        mutate_matrix(self.b2)

    def crossover(self, other):
        child = NeuralNetwork()

        # Utils function that return a mask with random bool
        def mix(a, b):
            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)

        child.w1 = mix(self.w1, other.w1)
        child.b1 = mix(self.b1, other.b1)
        child.w2 = mix(self.w2, other.w2)
        child.b2 = mix(self.b2, other.b2)
        return child


# ******** Utils ********
def generate_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def save_best_bird(bird):
    np.savez(SAVE_FILE,
             w1=bird.brain.w1,
             b1=bird.brain.b1,
             w2=bird.brain.w2,
             b2=bird.brain.b2
             )
    print("Goat has been saved")


def load_bird(filename=SAVE_FILE):
    data = np.load(filename)

    bird = Bird(100, WIN_HEIGHT // 2)
    bird.brain.w1 = data["w1"]
    bird.brain.b1 = data["b1"]
    bird.brain.w2 = data["w2"]
    bird.brain.b2 = data["b2"]

    print("Goat has been loaded")
    return bird


# ******** Classes ********
class Bird:
    def __init__(self, x, y, color=COLOR_GRAY):
        self.x = x
        self.y = y
        self.tick_count = 0
        self.vel = 0
        self.radius = 15
        self.alive = True
        self.jump_cooldown = 0
        self.color = color
        self.score = 0
        self.brain = NeuralNetwork()

    def jump(self):
        self.vel = JUMP_STRENGTH

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel
        self.jump_cooldown = max(0, self.jump_cooldown - 1)

        # Check for out of bounds
        if self.y > WIN_HEIGHT or self.y < 0:
            self.alive = False

    def think(self, pipe):
        if pipe is None:
            return

        top_y = (pipe.y - PIPE_GAP // 2) / WIN_HEIGHT
        bottom_y = (pipe.y + PIPE_GAP // 2) / WIN_HEIGHT
        center_gap_y = pipe.y / WIN_HEIGHT

        inputs = [self.y / WIN_HEIGHT,
                  (pipe.x - self.x) / WIN_WIDTH,
                  self.vel / 10.0,
                  center_gap_y,
                  (self.y - pipe.y) / WIN_HEIGHT]

        output = self.brain.predict(inputs)
        if output > 0.5 and self.jump_cooldown == 0:
            self.jump()
            self.jump_cooldown = JUMP_COOLDOWN

    def draw(self, win):
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap = PIPE_GAP
        self.y = random.randint(200,  WIN_HEIGHT - 200)

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
    def __init__(self, load_best=False):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Flappy UwU")
        self.clock = pygame.time.Clock()
        self.birds = [Bird(100, WIN_HEIGHT // 2)
                      for _ in range(BIRD_COUNT)]
        self.pipes = [Pipe(WIN_WIDTH + i * 300) for i in range(3)]
        self.score = 0
        self.generation = 1
        self.font = pygame.font.SysFont("jetbrainsmononlnfpmedium", 24)
        self.history_max = []
        self.history_mean = []

        if load_best and os.path.exists(SAVE_FILE):
            bird = load_bird()
            self.birds = [bird]
            self.pipes = [Pipe(WIN_WIDTH + i * 300) for i in range(3)]
            self.play_mode = True
        else:
            self.birds = [Bird(100, WIN_HEIGHT // 2)
                          for _ in range(BIRD_COUNT)]
            self.pipes = [Pipe(WIN_WIDTH + i * 300) for i in range(3)]
            self.play_mode = False

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

                # Reward for being alive
                bird.score += 0.1

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
            self.score += 2
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

        text_str = f"Score: {self.score}, Gen: {self.generation}"
        text = self.font.render(text_str, True, COLOR_WHITE)
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def evolve(self):
        # Saving score for graphs
        scores = [b.score for b in self.birds]
        self.history_max.append(max(scores))
        self.history_mean.append(np.mean(scores))

        # Sort by fitness
        sorted_birds = sorted(self.birds, key=lambda b: b.score, reverse=True)
        elite = sorted_birds[0]  # the GOAAAAT
        save_best_bird(elite)
        top_n = max(2, BIRD_COUNT // 5)  # 20%
        parents = sorted_birds[:top_n]

        new_birds = []
        # Keep the elite
        elite_child = Bird(100, WIN_HEIGHT // 2, COLOR_ELITE)
        elite_child.brain = elite.brain.copy()
        new_birds.append(elite_child)

        # Fill with children
        while len(new_birds) < BIRD_COUNT:
            if random.random() < 0.1:
                # 10% of new random bird
                child = Bird(100, WIN_HEIGHT//2)
            else:
                parent1 = random.choice(parents)
                # Crossover
                if random.random() < CROSSOVER_RATE:
                    parent2 = random.choice(parents)
                    child_brain = parent1.brain.crossover(parent2.brain)
                else:
                    child_brain = parent1.brain.copy()
                # Mutation
                child_brain.mutate(MUTATION_RATE)
                child = Bird(100, WIN_HEIGHT // 2)
                child.brain = child_brain

            new_birds.append(child)

        # Reset with new gen
        self.birds = new_birds
        self.pipes = [Pipe(WIN_WIDTH + i * 300) for i in range(3)]
        self.score = 0
        self.generation += 1

    def plot_progress(self):
        plt.style.use("seaborn-v0_8-pastel")
        plt.plot(self.history_max, label="Max Score")
        plt.plot(self.history_mean, label="Mean Score")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("Flappy UwU AI Progress")
        plt.legend()
        plt.grid(True, linestyle="--")
        plt.tight_layout()
        plt.show()

    def run(self):
        running = True

        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            alive_count = self.update()

            if not self.play_mode and alive_count == 0:
                self.evolve()
            self.draw()

        pygame.quit()
        if not self.play_mode:
            self.plot_progress()


if __name__ == "__main__":
    Game(load_best=False).run()

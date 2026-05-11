import pygame
import numpy as np
import random
import math
# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Brain Animation")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
NEURON_COLOR = (100, 200, 255)
SIGNAL_COLOR = (255, 255, 255)
LINE_COLOR = (30, 60, 80)

# Neurons configuration
NUM_NEURONS = 100
# Create a rough 3D brain shape (ellipsoid)
neurons = []
for _ in range(NUM_NEURONS):
    # Spherical to Cartesian coords for shape
    phi = random.uniform(0, 2 * math.pi)
    costheta = random.uniform(-1, 1)
    u = random.random()
    theta = math.acos(costheta)
    r = 150 * (u**(1/3))  # Radius
    
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi) * 0.6  # Flatten y
    z = r * math.cos(theta)
    neurons.append([x, y, z])

neurons = np.array(neurons)

# Generate connections (edges)
connections = []
for i in range(NUM_NEURONS):
    for j in range(i + 1, NUM_NEURONS):
        dist = np.linalg.norm(neurons[i] - neurons[j])
        if dist < 100:  # Only connect close neurons
            if random.random() < 0.2:
                connections.append((i, j))

# Signal class to animate firing
class Signal:
    def __init__(self, start_node, end_node):
        self.start = start_node
        self.end = end_node
        self.progress = 0
        self.speed = random.uniform(0.01, 0.03)

    def update(self):
        self.progress += self.speed
        if self.progress > 1:
            return False
        return True

    def get_pos(self):
        return self.start + (self.end - self.start) * self.progress

signals = []
angle = 0

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    
    # 3D Rotation
    angle += 0.01
    rotated_neurons = []
    for n in neurons:
        x, y, z = n
        # Rotate around Y axis
        new_x = x * math.cos(angle) - z * math.sin(angle)
        new_z = x * math.sin(angle) + z * math.cos(angle)
        rotated_neurons.append([new_x, y, new_z])
    
    # 3D to 2D Projection
    projected_points = []
    for n in rotated_neurons:
        factor = 400 / (400 + n[2]) # Perspective
        x = n[0] * factor + WIDTH / 2
        y = n[1] * factor + HEIGHT / 2
        projected_points.append((int(x), int(y)))

    # Draw lines (connections)
    for conn in connections:
        p1 = projected_points[conn[0]]
        p2 = projected_points[conn[1]]
        pygame.draw.line(screen, LINE_COLOR, p1, p2, 1)

    '''
import pygame
import random
import math

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors
BG_COLOR = (10, 10, 20)
NEURON_COLOR = (100, 200, 255)
SIGNAL_COLOR = (255, 255, 100)
LINE_COLOR = (30, 50, 70)

# Neural Network Structure
layers = [5, 8, 8, 5]
neurons = []
signals = []

# Generate neuron positions
def create_network():
    neurons.clear()
    layer_gap = WIDTH // (len(layers) + 1)
    for i, num_neurons in enumerate(layers):
        layer_x = (i + 1) * layer_gap
        col_neurons = []
        for j in range(num_neurons):
            neuron_y = int(HEIGHT / (num_neurons + 1) * (j + 1))
            col_neurons.append((layer_x, neuron_y))
        neurons.append(col_neurons)

# Signal Class
class Signal:
    def __init__(self, start_node, end_node):
        self.start = start_node
        self.end = end_node
        self.pos = list(start_node)
        self.speed = random.uniform(0.01, 0.03)
        self.t = 0 # progress

    def update(self):
        self.t += self.speed
        if self.t >= 1:
            return False
        # Linear interpolation for smooth movement
        self.pos[0] = self.start[0] + (self.end[0] - self.start[0]) * self.t
        self.pos[1] = self.start[1] + (self.end[1] - self.start[1]) * self.t
        return True

    def draw(self):
        pygame.draw.circle(screen, SIGNAL_COLOR, (int(self.pos[0]), int(self.pos[1])), 4)

create_network()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BG_COLOR)

    # Draw connections
    for i in range(len(neurons) - 1):
        for n1 in neurons[i]:
            for n2 in neurons[i+1]:
                pygame.draw.line(screen, LINE_COLOR, n1, n2, 1)

    # Spawn signals randomly
    if random.random() < 0.2:
        layer_idx = random.randint(0, len(layers) - 2)
        start_n = random.choice(neurons[layer_idx])
        end_n = random.choice(neurons[layer_idx + 1])
        signals.append(Signal(start_n, end_n))

    # Update and draw signals
    for signal in signals[:]:
        if not signal.update():
            signals.remove(signal)
        signal.draw()

    # Draw neurons
    for layer in neurons:
        for neuron in layer:
            pygame.draw.circle(screen, NEURON_COLOR, neuron, 10)
            pygame.draw.circle(screen, (0,0,0), neuron, 10, 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
'''
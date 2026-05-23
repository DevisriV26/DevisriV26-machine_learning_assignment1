import matplotlib.pyplot as plt
import time

# Initial state: 1 = dirty, 0 = clean
state = [1, 1]
pos = 0.0   # use float for smooth motion

# Policy
def choose_action(state, pos):
    current_room = int(round(pos))
    if state[current_room] == 1:
        return "suck"
    elif current_room == 0:
        return "right"
    else:
        return "left"

# Draw function
def draw(state, pos, step, action):
    plt.clf()

    for i in range(2):
        color = 'lightgreen' if state[i] == 0 else 'salmon'
        plt.gca().add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

        # Dirt
        if state[i] == 1:
            plt.text(i + 0.3, 0.6, '●', fontsize=16)
            plt.text(i + 0.6, 0.3, '●', fontsize=16)

    # Vacuum (smooth position)
    plt.text(pos + 0.3, 0.2, '🤖', fontsize=20)

    plt.xlim(0, 2)
    plt.ylim(0, 1)
    plt.title(f"Step {step} | Action: {action}")
    plt.xticks([])
    plt.yticks([])

    plt.pause(0.1)

# Smooth movement function
def move_smooth(start, end, state, step):
    global pos
    steps = 20
    for i in range(steps):
        pos = start + (end - start) * (i / steps)
        draw(state, pos, step, "moving")
        time.sleep(0.05)

# Simulation
plt.ion()

step = 0
while True:
    current_room = int(round(pos))
    action = choose_action(state, pos)

    if action == "suck":
        # show cleaning animation
        for _ in range(3):
            draw(state, pos, step, "cleaning...")
        state[current_room] = 0

    elif action == "right":
        move_smooth(0, 1, state, step)

    elif action == "left":
        move_smooth(1, 0, state, step)

    step += 1

    # Stop when clean
    if state == [0, 0]:
        draw(state, pos, step, "Finished ✅")
        break

plt.ioff()
plt.show()

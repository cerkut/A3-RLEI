import numpy as np
import tensorflow as tf

# ============================================
# 2D Gridworld Environment
# ============================================
class GridWorld:
    """
    N x N grid.
    Start at (0,0).
    Goal at (N-1, N-1).
    Reward +1 at goal, -1 at traps (optional), 0 otherwise.
    Episode ends at goal or after max steps.
    """

    def __init__(self, n=5):
        self.n = n
        self.goal = (n - 1, n - 1)
        self.max_steps = n * n

        # Optional traps
        self.traps = {(1, 2), (3, 1)}

    def reset(self):
        self.x = 0
        self.y = 0
        self.steps = 0
        return self._obs()

    def _obs(self):
        # Normalize to [0,1]
        return np.array([
            self.x / (self.n - 1),
            self.y / (self.n - 1)
        ], dtype=np.float32)

    def step(self, action):
        # 0 = left, 1 = right, 2 = up, 3 = down
        if action == 0:   # left
            self.x = max(0, self.x - 1)
        elif action == 1: # right
            self.x = min(self.n - 1, self.x + 1)
        elif action == 2: # up
            self.y = max(0, self.y - 1)
        elif action == 3: # down
            self.y = min(self.n - 1, self.y + 1)

        self.steps += 1
        done = False

        if (self.x, self.y) == self.goal:
            return self._obs(), 1.0, True

        if (self.x, self.y) in self.traps:
            return self._obs(), -1.0, True

        if self.steps >= self.max_steps:
            return self._obs(), 0.0, True

        return self._obs(), 0.0, False


# ============================================
# Q-network (small enough for Edge Impulse)
# ============================================
def build_q_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),    # x_norm, y_norm
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4)              # 4 actions
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse"
    )
    return model


# ============================================
# Q-learning
# ============================================
def train_qlearning(episodes=1500, gamma=0.99):
    env = GridWorld(n=5)
    model = build_q_model()

    epsilon = 1.0
    eps_min = 0.05
    eps_decay = 0.995

    for ep in range(episodes):
        s = env.reset()
        done = False

        while not done:
            if np.random.rand() < epsilon:
                a = np.random.randint(4)
            else:
                q = model.predict(s.reshape(1, 2), verbose=0)[0]
                a = int(np.argmax(q))

            ns, r, done = env.step(a)

            q_curr = model.predict(s.reshape(1, 2), verbose=0)[0]
            if done:
                target = r
            else:
                q_next = model.predict(ns.reshape(1, 2), verbose=0)[0]
                target = r + gamma * np.max(q_next)

            q_target = q_curr.copy()
            q_target[a] = target

            model.fit(
                s.reshape(1, 2),
                q_target.reshape(1, 4),
                verbose=0
            )

            s = ns

        epsilon = max(eps_min, epsilon * eps_decay)
        if ep % 100 == 0:
            print(f"Episode {ep}, epsilon={epsilon:.3f}")

    return model


# ============================================
# Export for Edge Impulse
# ============================================
if __name__ == "__main__":
    model = train_qlearning()

    model.save("grid_q_model.h5")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = conv.convert()

    with open("grid_q_model.tflite", "wb") as f:
        f.write(tflite)

    print("Exported grid_q_model.tflite")
    
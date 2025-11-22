import numpy as np
import tensorflow as tf

# ============================================
# Simple 1-D environment for demonstration
# ============================================
class LineWorld:
    """
    States = 0...(n_states-1).
    Goal: move right to +1 reward, left = -1 reward.
    Observation passed to the network = [state_normalized].
    """
    def __init__(self, n_states=7):
        self.n_states = n_states
        self.start = n_states // 2

    def reset(self):
        self.state = self.start
        return np.array([self.state / (self.n_states - 1)], dtype=np.float32)

    def step(self, action):
        # 0 = left, 1 = right
        self.state += -1 if action == 0 else +1
        self.state = max(0, min(self.state, self.n_states - 1))

        done = False
        reward = 0.0

        if self.state == 0:
            reward = -1.0
            done = True
        elif self.state == self.n_states - 1:
            reward = +1.0
            done = True

        return (
            np.array([self.state / (self.n_states - 1)], dtype=np.float32),
            reward,
            done
        )

# ============================================
# Tiny Q-network suitable for Edge Impulse
# ============================================
def build_q_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),   # scalar input
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(2)             # Q-values for [left, right]
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse"
    )
    return model

# ============================================
# Q-learning training loop
# ============================================
def train_qlearning(episodes=500, gamma=0.95):
    env = LineWorld()
    model = build_q_model()

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    for ep in range(episodes):
        s = env.reset()
        done = False

        while not done:
            # epsilon-greedy action
            if np.random.rand() < epsilon:
                a = np.random.randint(2)
            else:
                q = model.predict(s.reshape(1, 1), verbose=0)
                a = int(np.argmax(q))

            ns, r, done = env.step(a)

            # compute target Q
            q_current = model.predict(s.reshape(1, 1), verbose=0)[0]

            if done:
                target_q = r
            else:
                q_next = model.predict(ns.reshape(1, 1), verbose=0)[0]
                target_q = r + gamma * np.max(q_next)

            # update only the selected action
            q_target_all = q_current.copy()
            q_target_all[a] = target_q

            model.fit(
                s.reshape(1, 1),
                q_target_all.reshape(1, 2),
                verbose=0
            )

            s = ns

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 50 == 0:
            print(f"Episode {ep}, epsilon={epsilon:.3f}")

    return model

# ============================================
# Train + export for Edge Impulse Studio
# ============================================
if __name__ == "__main__":
    model = train_qlearning()

    # Save Keras model
    model.save("q_model.h5")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = converter.convert()

    with open("q_model.tflite", "wb") as f:
        f.write(tflite)

    print("Exported q_model.tflite for Edge Impulse.")
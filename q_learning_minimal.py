import numpy as np
import tensorflow as tf

# ============================
# Simple 1D LineWorld Environment
# ============================
class LineWorld:
    """
    States: positions 0...(n_states-1)
    Start in middle.
    Action 0 = left, 1 = right.
    State 0  -> terminal, reward -1
    State n-1 -> terminal, reward +1
    All others reward 0.
    """
    def __init__(self, n_states=7):
        self.n_states = n_states
        self.start_state = n_states // 2
        self.reset()

    def reset(self):
        self.state = self.start_state
        return self._obs()

    def _obs(self):
        # scalar observation normalized to [0,1]
        return np.array([self.state / (self.n_states - 1)], dtype=np.float32)

    def step(self, action):
        # action: 0 = left, 1 = right
        if action == 0:
            self.state -= 1
        else:
            self.state += 1

        # clip to bounds
        self.state = max(0, min(self.state, self.n_states - 1))

        # rewards and done
        if self.state == 0:
            reward = -1.0
            done = True
        elif self.state == self.n_states - 1:
            reward = +1.0
            done = True
        else:
            reward = 0.0
            done = False

        return self._obs(), reward, done


# ============================
# Q-network in TensorFlow / Keras
# ============================
def create_q_model(n_actions=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),      # scalar position
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(n_actions)        # Q-values for each action
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    return model


# ============================
# Q-learning loop
# ============================
def train_q_learning(
    n_episodes=500,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
):
    env = LineWorld(n_states=7)
    n_actions = 2
    model = create_q_model(n_actions=n_actions)

    epsilon = epsilon_start

    for episode in range(n_episodes):
        state = env.reset()  # shape (1,)
        done = False
        total_reward = 0.0

        while not done:
            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                q_values = model.predict(state[None, :], verbose=0)[0]
                action = int(np.argmax(q_values))

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Compute target Q value
            q_values = model.predict(state[None, :], verbose=0)[0]
            if done:
                target = reward
            else:
                next_q_values = model.predict(next_state[None, :], verbose=0)[0]
                target = reward + gamma * np.max(next_q_values)

            # Update only the chosen action’s Q-value
            q_values_target = q_values.copy()
            q_values_target[action] = target

            # Fit model on this single (state, target_q_vector)
            model.fit(state[None, :],
                      q_values_target[None, :],
                      verbose=0)

            state = next_state

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{n_episodes}, "
                  f"epsilon={epsilon:.3f}, total_reward={total_reward:.1f}")

    return model


if __name__ == "__main__":
    model = train_q_learning()

    # ============================
    # Save for Edge Impulse
    # ============================
       # Build a serving model that returns both Q and softmax probabilities
    x_in = tf.keras.Input(shape=(1,), name="x")                 # EI feature vector (1 float)
    q_out = model(x_in, training=False)
    prob_out = tf.keras.layers.Softmax(name="probabilities")(q_out)
    serving_model = tf.keras.Model(inputs=x_in, outputs={"q": q_out, "probabilities": prob_out})

    # Example: get x from EI (here we just simulate x=0.8)
    # x = np.array([[0.8]], dtype=np.float32)
    # outs = serving_model.predict(x, verbose=0)
    # print("Q:", outs["q"][0], "probs:", outs["probabilities"][0], "sum=", outs["probabilities"][0].sum())

    # Save single-output Q model (optional)
    # model.save("q_lineworld_model.h5")

    # Save two-output serving model (Keras + TFLite)
    # serving_model.save("q_lineworld_serving.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(serving_model)
    tflite_model = converter.convert()
    with open("q_lineworld_serving.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("Saved q_lineworld_serving.tflite")
# A3-RLEI

TensorFlow-based Q-learning models integrated into Edge Impulse Studio using BYOM.

3 models are used: 1) Basic 1D Model for Arduino Nano BL-33, 2) Minimal Generic 1D model, and 3) 2D Grid Model.

![gridworld_loop0.gif](https://usercdn.edgeimpulse.com/project828999/090d49d92168d3a633b60ef6da015ea9e0b9134a3635b887793180c507b19d66)

The *value* of our apporach can be summarized as follows:

✔ Proof that RL policies can run on microcontrollers: The Q-network is tiny, fast, and perfectly suited for MCU execution.

✔ Demonstrates merging RL + TinyML + Edge Impulse: Students see how to pipe an RL-trained network through the Edge Impulse toolchain.

✔ Enables adaptive embedded agents: Decision-making becomes learned and optimized rather than hard-coded.

✔ Fits real industrial applications: Where RL is trained offline and deployed as a compact model.

✔ Shows a general method: Any RL policy (DQN, PPO actor, etc.) can be converted to TFLite and uploaded.

# Application Scenarios 1D

## 1. Application Scenario: “Adaptive Embedded Agent”

A tiny embedded device (e.g., Nano 33 BLE) operates in a simple but uncertain environment — for example, choosing actions based on proximity, position, or a single sensor reading.

Instead of a fixed ruleset, the system learns a policy (via Q-learning) that can be deployed as a lightweight neural network.

### Examples

    •	A robot that must move left or right to reach a target zone.
	•	A toy vehicle that avoids getting stuck by choosing the best direction based on simple distance sensors.
	•	A small embedded system that regulates a single measurable variable (e.g. temperature offset, wheel position, hinge angle), taking “left/right” or “increase/decrease” actions.

### Value

    •	Replaces manually-coded rules (“if distance < 10 cm then move left”) with a learned control strategy.
	•	Runs in real time on ultra-low-power hardware using a tiny TFLite model.
	•	Demonstrates how reinforcement learning models can be compressed and executed like any other TinyML model.
	•	Makes behavior consistent, efficient, and data-driven, which is powerful for educational and prototyping use.

## 2. Application Scenario: “One-Dimensional State Decision-Maker”

This model is perfect whenever our system measures one scalar, then chooses between two actions.

### Examples

    •	Edge-based thermostat bumping: If temperature deviation is positive/negative → decide to heat or cool, but learned via reward dynamics.
	•	Gesture-driven binary decision: If distance sensor detects the user’s hand at a position, the model chooses “left function” or “right function.”
	•	Line-following educational robot:
        - State = current horizontal offset from center of a line;
        - Action = left/right wheel speed correction.

### Value

    •	Turns a simple 1D input into optimal action selection.
	•	Easy to test inside Edge Impulse Studio (just feed values 0–1).
	•	Opens doors for behavior personalization:

Train on our driving/steering pattern → deploy on device.

## 3. Application Scenario: “Offline RL Policy Deployment for TinyML”

Edge Impulse does not run RL loops directly — but we can deploy the trained policy.

### Scenario

We train the agent externally; Edge Impulse runs inference only.
This mimics real industrial workflows:

    •	We train an RL agent using simulations, historical logs, or user demonstrations
	•	Compress to TFLite
	•	Deploy to embedded devices in the field

### Industry parallels

    •	HVAC controllers learning optimal heating policies
	•	Motor control systems learning optimal torque adjustments
	•	User-adaptive wearable behaviors (e.g., adjusting sound modes based on user actions)
	•	Small robots reacting to single-sensor signals

### Value

    •	Demonstrates how reinforcement learning policies can be converted into TinyML models, enabling:
	•	ultra-low-latency decisions
	•	offline execution
	•	extremely low energy consumption
	•	Fits naturally into Edge Impulse’s deployment ecosystem.

## 4. Application Scenario: “Educational RL-to-TinyML Pipeline”

This is the most important value for teaching and prototyping:

### Scenario

Students build a simple RL agent, train it, export it, and run it inside Edge Impulse — experiencing the whole pipeline from training → model → embedded inference.

### Value

    •	Makes RL tangible for embedded students.
	•	Shows how RL policies can be integrated like any normal model.
	•	Exercises full ML lifecycle: training → conversion → inference → deployment.
	•	Bridges AI (RL) with embedded systems and Edge Impulse workflows.

# 🎮 2D Gridworld Scenario

A small agent moves on a 2D grid. The **goal** is to reach a target cell while avoiding penalty cells.
The observation becomes a 2-feature vector:

[x_position_normalized, y_position_normalized]

The Q-network outputs four Q-values:

[left, right, up, down]

## Application Scenario

Imagine a robot navigating a small floor plan.
The grid abstracts obstacles and safe zones.
Once trained, the policy runs on a Nano 33 BLE with almost no power.
Inference is instant.
This gives a learned controller without heavy computation.

This is now a full navigation agent compressed into a tiny network.
It is realistic and teaches:

    •	RL reward shaping
	•	State encoding
	•	Policy extraction
	•	TinyML deployment
	•	How Q-networks replace hand-written control logic

It also acts as a foundation for more advanced tasks:

    •	visual navigation
	•	mobile robot control
	•	sensor-driven grid mapping
	•	multi-goal routing strategies

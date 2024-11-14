import gymnasium as gym
from gymnasium import spaces
from mappings.block_mappings import BLOCK_MAPPINGS
from mappings.reward_mappings import REWARD_MAPPINGS
import numpy as np
import socket
import json
import time

actions = ["turn-right", "turn-left", "move-forward", "mine", "tilt-down", "tilt-up", "forward-up", "forward-down"]
directions = ["targetBlock", "behindTop", "behindBottom", "leftTop", "forwardTop", "rightBottom","leftBottom", "up", "forwardBottom", "down", "rightTop", "forwardAbove", "forwardBelow"]



class MinecraftAgentEnv(gym.Env):
    def __init__(self, host="localhost", port=5000, max_steps=1024):
        super(MinecraftAgentEnv, self).__init__()

        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self._start_server()

        # Shifted coordinate space
        x_max, y_max, z_max = 128, 62, 128
        self.coordinate_space = spaces.MultiDiscrete([x_max + 1, y_max + 1, z_max + 1])

        tilt_values = [-90,-45,0,45,90]
        self.tilt_space = spaces.Discrete(len(tilt_values))

        # Surrounding block space (12 blocks surrounding the player + 1 target block)
        num_block_types = len(BLOCK_MAPPINGS)
        self.surrounding_block_space = spaces.MultiDiscrete([num_block_types] * len(directions))

        # Observation space as a combination of coordinates and surrounding blocks
        self.observation_space = spaces.Tuple((
            self.coordinate_space,
            self.surrounding_block_space,
            self.tilt_space
        ))

        # Action space: 8 discrete actions (turn-left, turn-right, forward, tilt-up, tilt-down, forward-up, forward-down and mine)
        self.action_space = spaces.Discrete(8)

        self.max_steps = max_steps
        self.step_count = 0

    def _start_server(self):
        # Sets up the server socket, waiting for the plugin to connect using /connect
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")

        self.client_socket, addr = self.server_socket.accept()
        print(f"Client connected: {addr}")

    

    def reset(self):
        # Consider whether best approach is for model to issue reset, 
        # or if plugin should keep track of the amount of actions performed and calls reset after a given amount.
        self.client_socket.sendall(b"RESET\n")
        initial_state, result = self._receive_state()
    
        self.step_count = 0
        return initial_state, {}
    
    def step(self, action):
        # Sends action to plugin
        action_str = actions[action]
        #time.sleep(0.1)
        self.client_socket.sendall(f"{action_str}\n".encode('utf-8'))

        state, result = self._receive_state()
        #print(f"Result: {result}")

        reward = self._calculate_reward(result)
        #print(f"Reward: {reward}")

        self.step_count += 1

        done = self.step_count >= self.max_steps

        return state, reward, done, result

    def _receive_state(self):
        state_data = self.client_socket.recv(1024).decode('utf-8').strip()

        try:
            state_json = json.loads(state_data)

            x,y,z = state_json.get("x", 0), state_json.get("y", 0), state_json.get("z", 0)
            encoded_coordinates = np.array([x,y,z], dtype=np.float32)

            tilt = state_json.get("tilt", 0)

            action_result = state_json.get("actionResult", "unknown")
            print(action_result)

            surrounding_blocks_dict = state_json.get("surroundingBlocks", {})

            surrounding_blocks = [
                BLOCK_MAPPINGS.get(surrounding_blocks_dict.get(direction, "AIR"), 0) 
                for direction in directions
            ]
            """
            observation = {
                "coordinates": encoded_coordinates,
                "tilt": tilt,
                "surrounding_blocks": np.array(surrounding_blocks, )
            }"""
            state = np.concatenate([encoded_coordinates, surrounding_blocks, [tilt]])
            return state, action_result
        except json.JSONDecodeError:
            print("Failed to decode json from plugin")
            return self.observation_space.sample()
        
    def close(self):
        if self.client_socket:
            self.client_socket.close()
            print("Client disconnected")
        if self.server_socket:
            self.server_socket.close()
            print("Server closed")

    def _calculate_reward(self, result):
        return REWARD_MAPPINGS.get(result, REWARD_MAPPINGS.get("unknown", 0))




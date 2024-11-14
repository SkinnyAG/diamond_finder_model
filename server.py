import socket
import json
import random

actions = ["turn-right", "turn-left", "move-forward", "mine", "tilt-down", "tilt-up", "forward-up", "forward-down"]

class Model:
    def predict(self, state):
        action = random.choice(actions)
        return action
    

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 5000))
    server.listen(1)

    print("Server listening on port 5000")

    client_socket, addr = server.accept()
    print(f"Client connected: {addr}")

    model = Model()

    try:
        while True:
            state = client_socket.recv(1024).decode('utf-8')
            if not state:
                break

            print(f"Received state: {state}")

            try:
                state_data = json.loads(state)
            except json.JSONDecodeError:
                print("Received malformed JSON. Skipping")
                continue

            action = model.predict(state_data)
            print(f"Predicted action: {action}")

            response = action + "\n"
            client_socket.sendall(response.encode('utf-8'))
    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        client_socket.close()
        print("Client disconnected")
        server.close()


if __name__ == "__main__":
    start_server()
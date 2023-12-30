from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import threading
import torch

class TrainingServer:
    def __init__(self):
        self.osc_input_port = 8000  # Set the port to listen to
        self.training_data_dict = {}
        self.init_osc_server()

    def init_osc_server(self):
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/training_data", self.handle_training_data)
        self.osc_ip = "0.0.0.0"  # Listening on all interfaces
        self.osc_port = self.osc_input_port  # Port 8000
        self.osc_server = BlockingOSCUDPServer((self.osc_ip, self.osc_port), self.dispatcher)

    def start_osc_server(self):
        """Starts the OSC server in a separate thread."""
        self.server_thread = threading.Thread(target=self.osc_server.serve_forever)
        self.server_thread.start()
        print("OSC Server started on a new thread")

    def stop_osc_server(self):
        if self.osc_server:
            self.osc_server.shutdown()
            self.server_thread.join()  # Ensure the server thread has fully terminated
            print("OSC Server stopped")

    def handle_training_data(self, address, *args):
        pose_count, audio_buffer_count = args[0], args[1]
        logits_len = int(args[2])
        # Calculate start and end indices for each segment
        logits_start = 3  # Starting after pose_count, audio_buffer_count, logits_len
        logits_end = logits_start + logits_len
        class_prediction_start = logits_end + 1
        class_prediction_end = class_prediction_start + int(args[logits_end])
        softmax_values_start = class_prediction_end + 1
        softmax_values_end = softmax_values_start + int(args[class_prediction_end])
        input_audio_start = softmax_values_end + 1
        input_audio_end = input_audio_start + int(args[softmax_values_end])

        # Convert the slices of remaining to tensors
        logits_tensor = torch.tensor(args[logits_start:logits_end])
        class_prediction_tensor = torch.tensor(args[class_prediction_start:class_prediction_end])
        softmax_values_tensor = torch.tensor(args[softmax_values_start:softmax_values_end])
        input_audio_tensor = torch.tensor(args[input_audio_start:input_audio_end]).reshape(1, 16, -1)

        # Efficiently get or create the list for audio_buffer_count
        tensors_list = self.training_data_dict.setdefault(audio_buffer_count, [])

        # Append the new data
        tensors_list.append([
            pose_count, logits_tensor, class_prediction_tensor,
            softmax_values_tensor, input_audio_tensor
        ])
        #print(tensors_list)
    
    def get_next_training_batch(self):
        pass
    
    def get_tensor_data(self, audio_buffer_count, index):
        """Retrieve tensor data for a specific audio_buffer_count and index."""
        # Check if the specified audio_buffer_count is in the training data dictionary
        if audio_buffer_count in self.training_data_dict:
            tensor_list = self.training_data_dict[audio_buffer_count]
            # Check if the specified index is within the range of the tensor list
            if 0 <= index < len(tensor_list):
                return tensor_list[index]
            else:
                raise IndexError("Specified index is out of range for the audio_buffer_count.")
        else:
            raise KeyError(f"No data for audio_buffer_count: {audio_buffer_count}")


if __name__ == "__main__":
    server = TrainingServer()  # Create an instance of TrainingServer
    try:
        server.start_osc_server()
    except KeyboardInterrupt:
        server.stop_osc_server()  # Stop the server when KeyboardInterrupt is caught
        # After stopping the server, attempt to retrieve and print the data
        try:
            data = server.get_tensor_data(0, 0)  # Use the instance 'server' to call get_tensor_data
            print(data)
        except (KeyError, IndexError) as e:
            print(f"Error retrieving data: {e}")
        print("Server stopped by user")

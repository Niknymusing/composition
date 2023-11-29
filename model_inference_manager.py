import torch
import threading
import time
from math import prod

class ModelManager:
    def __init__(self, model_class, *model_args):
        self.model1 = model_class(*model_args)
        self.model2 = model_class(*model_args)
        self.model1.eval()
        self.model2.eval()
        self.lock = threading.Lock() 
        self.use_model1 = False  # Flag to indicate which model is currently active
        self.switch_time = time.time()


    def update_model(self, model, updated_params):
        # Update the specified model with the received parameters
        with torch.no_grad():
            for param, new_param in zip(model.parameters(), updated_params):
                param.data.copy_(new_param)

    def get_active_model(self):
        # Return the currently active model
        return self.model1 if self.use_model1 else self.model2

    def switch_models(self):
        # Switch the active model
        print('switch model, time since last update = ', time.time() - self.switch_time)
        self.use_model1 = not self.use_model1
        self.switch_time = time.time()


    def update_parameters(self, parameter_values):
        
        with self.lock:
            model = self.get_active_model()
            self.switch_models()
            if len(list(model.parameters())) != len(parameter_values):
                #raise ValueError("The number of parameters and provided values do not match.")
                print('value error')
            
            sorted_indices = sorted(list(parameter_values.keys()))
            for param, i in zip(model.parameters(), sorted_indices):
                shape = parameter_values[i][2]
                #print(f"Processing parameter {i}, shape: {shape}")

                # Process each chunk
                tensor_chunks = []
                for chunk in parameter_values[i][0][:-1]:
                    if isinstance(chunk, list):
                        tensor_chunk = torch.tensor(chunk).flatten()
                    else:
                        print(f"Non-list chunk encountered: {chunk}, type: {type(chunk)}")
                        tensor_chunk = torch.tensor([chunk])

                    tensor_chunks.append(tensor_chunk)

                # Handle the last chunk if the parameter was split into chunks
                if parameter_values[i][3] and isinstance(parameter_values[i][0][-1], list):
                    last_chunk = torch.tensor(parameter_values[i][0][-1]).flatten()
                    tensor_chunks.append(last_chunk)
                elif parameter_values[i][3]:
                    last_chunk = torch.tensor([parameter_values[i][0][-1]])
                    tensor_chunks.append(last_chunk)

                # Concatenate chunks
                if tensor_chunks:
                    values = torch.cat(tensor_chunks)
                    if len(values) == prod(shape):
                        new_param = values.view(*shape)
                        param.data.copy_(new_param)
                        #print(f"Param {i} updated successfully")
                    else:
                        print(f"Mismatch in expected size for param {i}")#: Expected {prod(shape)}, got {len(values)}")
                else:
                    print(f"No chunks to process for param {i}")
        self.switch_models()
        print('Model parameters updated successfully')


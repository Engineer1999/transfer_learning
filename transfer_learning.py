import ultralytics
from ultralytics import YOLO
import torch

model = YOLO('yolov8s.pt')
freeze = [f"model.{x}." for x in range(8)]
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    if any(x in k for x in freeze):
        print(f"freezing {k}")
        v.requires_grad = False
        
for k , v in model.named_parameters():
    print(k, v.requires_grad)
    

train_data_path = './Package V2.v5i.yolov8/data.yaml'
epochs = 50

# Correctly format the training arguments
train_args = {
    'data': train_data_path,
    'epochs': epochs,
    'device': [0,1]
}
# Check the number of GPUs available
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# Train the model
model.train(**train_args)
import torch

# Load your model, mapping to CPU
model = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device("cpu"))

# Switch to evaluation mode
model.eval()

# Prepare the model for quantization
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare(model, inplace=True)

# Calibrate the model (you may want to pass a sample input through the model here)
# This step usually requires some data to calibrate the activations

# Convert the model to a quantized version
torch.quantization.convert(model, inplace=True)

# Save the quantized model
torch.save(model.state_dict(), "quantized_model.pth")

from flask import (
    Flask,
    request,
    jsonify,
)  # Import Flask framework and utility functions for handling requests and JSON responses
from PIL import Image  # Import Python Imaging Library for handling image operations
import io  # Import io for handling byte I/O operations
import torch  # Import PyTorch for building and using neural networks
import torch.nn.functional as F  # Import functional API for neural network operations
from facenet_pytorch import (
    MTCNN,
    InceptionResnetV1,
)  # Import face detection and recognition models from facenet_pytorch library
import cv2  # Import OpenCV for computer vision operations
from pytorch_grad_cam import (
    GradCAM,
)  # Import Grad-CAM for visualizing neural network activations
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)  # Utility for specifying target classes for Grad-CAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)  # Utility for overlaying Grad-CAM heatmap on an image
from flask_cors import (
    CORS,
)  # Import Cross-Origin Resource Sharing (CORS) support for Flask

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)  # Enable CORS for the app, allowing cross-origin requests

# Set the device to GPU if available, otherwise use CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize the MTCNN (Multi-task Cascaded Convolutional Networks) for face detection
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()

# Initialize the InceptionResnetV1 model with pre-trained weights for face classification (VGGFace2)
model = InceptionResnetV1(
    pretrained="vggface2", classify=True, num_classes=1, device=DEVICE
)

# Load a custom checkpoint for the model and set its state
checkpoint = torch.load(
    "resnetinceptionv1_epoch_32.pth", map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)  # Move the model to the appropriate device (GPU/CPU)
model.eval()  # Set the model to evaluation mode


def run_deepfake_detection(input_image: Image.Image):
    """
    Run deepfake detection on the input image and return the result.

    Args:
        input_image (PIL.Image): The input image.

    Returns:
        confidences (dict): Prediction confidence for 'real' and 'fake'.
        face_with_mask (numpy array): Image with the heatmap (GradCAM) applied.
    """
    # Detect face in the input image using MTCNN
    face = mtcnn(input_image)
    if face is None:
        raise Exception("No face detected")  # Raise an exception if no face is detected

    # Add a batch dimension to the detected face tensor and resize to 256x256
    face = face.unsqueeze(0)  # Shape: [1, 3, height, width]
    face = F.interpolate(face, size=(256, 256), mode="bilinear", align_corners=False)

    # Convert face tensor to a numpy array for visualization
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype(
        "uint8"
    )  # Ensure array is in uint8 format for image operations

    # Prepare the face tensor for model input
    face = face.to(DEVICE)  # Move tensor to the appropriate device
    face = face.to(torch.float32)  # Ensure tensor is in float32 format
    face = face / 255.0  # Normalize the pixel values to [0, 1]

    # Convert the preprocessed face tensor back to a numpy array for visualization
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    # Select the last layer of the model's block8 for Grad-CAM visualization
    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)  # Initialize Grad-CAM
    targets = [
        ClassifierOutputTarget(0)
    ]  # Define target output class for Grad-CAM (class 0)

    # Generate Grad-CAM heatmap for the face tensor
    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[
        0, :
    ]  # Extract the heatmap for the first batch element

    # Overlay the Grad-CAM heatmap on the original image
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(
        prev_face, 1, visualization, 0.5, 0
    )  # Combine original and heatmap images

    # Perform inference using the model to predict 'real' or 'fake'
    with torch.no_grad():
        output = torch.sigmoid(
            model(face).squeeze(0)
        )  # Get model output and apply sigmoid for probability
        prediction = (
            "real" if output.item() < 0.5 else "fake"
        )  # Classify based on output value

        # Calculate confidence scores for 'real' and 'fake'
        real_prediction = 1 - output.item()
        fake_prediction = output.item()

        # Store confidence scores in a dictionary
        confidences = {"real": real_prediction, "fake": fake_prediction}

    return (
        confidences,
        face_with_mask,
    )  # Return the confidence scores and visualization image


@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image file is included in the request
    if "file" not in request.files:
        return (
            jsonify({"error": "No file part"}),
            400,
        )  # Return error if no file is found in the request

    file = request.files["file"]  # Get the uploaded file

    # Check if the filename is not empty
    if file.filename == "":
        return (
            jsonify({"error": "No selected file"}),
            400,
        )  # Return error if no file is selected

    try:
        # Read the image file and convert it to RGB format
        input_image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Run deepfake detection and get the results
        confidences, face_with_mask = run_deepfake_detection(input_image)

        # Encode the visualization image to a byte stream for sending as a response
        _, buffer = cv2.imencode(".jpg", face_with_mask)
        face_with_mask_bytes = buffer.tobytes()

        # Return the prediction and confidence scores as a JSON response
        return jsonify(
            {
                "prediction": (
                    "fake" if confidences["fake"] > confidences["real"] else "real"
                ),
                "confidence_real": confidences["real"],
                "confidence_fake": confidences["fake"],
            }
        )

    except Exception as e:
        return (
            jsonify({"error": str(e)}),
            500,
        )  # Return error response if any exception occurs


# Run the Flask app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for the Flask app

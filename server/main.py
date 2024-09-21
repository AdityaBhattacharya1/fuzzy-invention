from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2", classify=True, num_classes=1, device=DEVICE
)

checkpoint = torch.load(
    "resnetinceptionv1_epoch_32.pth", map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


def run_deepfake_detection(input_image: Image.Image):
    """
    Run deepfake detection on the input image and return the result.

    Args:
        input_image (PIL.Image): The input image.

    Returns:
        confidences (dict): Prediction confidence for 'real' and 'fake'.
        face_with_mask (numpy array): Image with the heatmap (GradCAM) applied.
    """
    # Detect face in the input image
    face = mtcnn(input_image)
    if face is None:
        raise Exception("No face detected")

    # Preprocess the face
    face = face.unsqueeze(0)  # Add batch dimension
    face = F.interpolate(face, size=(256, 256), mode="bilinear", align_corners=False)

    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype("uint8")

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0

    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"

        # Confidence scores
        real_prediction = 1 - output.item()
        fake_prediction = output.item()

        confidences = {"real": real_prediction, "fake": fake_prediction}

    return confidences, face_with_mask


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        input_image = Image.open(io.BytesIO(file.read())).convert("RGB")

        confidences, face_with_mask = run_deepfake_detection(input_image)

        _, buffer = cv2.imencode(".jpg", face_with_mask)
        face_with_mask_bytes = buffer.tobytes()

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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

# Deepfake Detection App

This project is a Deepfake Detection application. It consists of a **Flask backend** (for detecting deepfakes in images) and a **Next.js frontend** (for users to upload images and get results). The frontend includes user authentication using Firebase Google OAuth.

## Table of Contents

1. [Backend](#backend)
    - [Technologies](#backend-technologies)
    - [Setup](#backend-setup)
    - [API Explanation](#backend-api-explanation)
2. [Frontend](#frontend)
    - [Technologies](#frontend-technologies)
    - [Setup](#frontend-setup)
    - [Protected Routes and Authentication](#protected-routes-and-authentication)
    - [Uploading Files](#uploading-files)
    - [Displaying Results](#displaying-results)

---

## Backend

### Backend Technologies

-   **Python**: The programming language used.
-   **Flask**: A lightweight web framework to handle HTTP requests.
-   **Torch**: For deep learning model execution.
-   **FaceNet-PyTorch**: For face detection and recognition.
-   **GradCAM**: For generating heatmaps over detected faces.
-   **Pillow**: For image handling and processing.

### Backend Setup

1. **Install Python dependencies**:
   Make sure you have Python installed. Create a virtual environment and install the necessary packages:

    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain all the necessary dependencies:

    ```
    Flask
    torch
    torchvision
    facenet-pytorch
    opencv-python
    pytorch-grad-cam
    pillow
    ```

2. **Run the Flask server**:

    ```bash
    python app.py
    ```

    This will start your backend server at `http://localhost:5000`.

### Backend API Explanation

The backend has a single route:

-   **POST /predict**:
    -   This route accepts an image uploaded via a form.
    -   It runs deepfake detection on the face detected in the image.
    -   The deepfake detection model returns a result with confidence values for **Real** and **Fake** faces.

The deepfake detection model is based on **InceptionResnetV1**, and it uses **GradCAM** to visualize where the model is "looking" when making predictions. The API responds with:

-   **Prediction**: Whether the face is "Real" or "Fake".
-   **Confidence Values**: Probability percentages for both "Real" and "Fake".

---

## Frontend

### Frontend Technologies

-   **Next.js**: The React framework used for building the frontend.
-   **Tailwind CSS**: For styling and UI design.
-   **DaisyUI**: A set of UI components built on top of Tailwind for rapid development.
-   **Firebase Authentication**: For Google OAuth login.
-   **Chart.js**: For creating charts (like pie charts) to display deepfake detection results.

### Frontend Setup

1. **Install Node.js dependencies**:

    Run this command in the frontend directory:

    ```bash
    npm install
    ```

2. **Run the Next.js frontend**:

    ```bash
    npm run dev
    ```

    The app will be available at `http://localhost:3000`.

### Protected Routes and Authentication

-   The app has two main pages:

    1. **Login Page** (`/login`): This page uses Firebase Google OAuth to allow users to sign in.
    2. **File Upload Page** (`/upload`): A protected route that can only be accessed if the user is logged in.

-   **How Authentication Works**:
    -   When a user clicks "Sign In with Google", they are redirected to Google to sign in.
    -   After signing in, they are redirected to the `/upload` page, where they can upload an image.
    -   The `onAuthStateChanged` hook from **Firebase Authentication** is used to track the user session. If no user is logged in, they are automatically redirected to the `/login` page when trying to access protected routes like `/upload`.

### Uploading Files

On the **Upload Page** (`/upload`):

-   The user can select an image file from their device.
-   When they click "Upload and Analyze", the file is sent to the backend API (`/predict`).
-   The backend processes the image, detects faces, and determines if the face is real or fake.

### Displaying Results

Once the backend returns the results:

-   The frontend displays the prediction: **Real** or **Fake**.
-   A **Pie Chart** is generated using **Chart.js** to show the confidence values (percentages for "Real" and "Fake").

---

## Conclusion

This project allows users to upload images and detect whether the face in the image is real or fake using deep learning models. The app is secured with Google OAuth login, and the results are displayed in a user-friendly way with charts and confidence percentages. Both the backend and frontend work together to create a smooth user experience.

---

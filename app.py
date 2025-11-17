import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- 1. Load MTCNN for face detection ---
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

# --- 2. Load the pre-trained Deepfake detection model ---
model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# --- 3. Helper function to process a single frame ---
def predict_frame(input_image: Image.Image):
    """
    Helper function to predict a single frame (as a PIL Image)
    Returns the 'fake' probability (float) or None if no face is detected.
    """
    try:
        face = mtcnn(input_image)
        if face is None:
            return None  # No face detected in this frame
        
        face = face.unsqueeze(0)  # add the batch dimension
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

        face = face.to(DEVICE)
        face = face.to(torch.float32)
        face = face / 255.0

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            return output.item()  # Return the 'fake' probability
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

# --- 4. Main function for the Gradio interface ---
def predict_video(video_path: str):
    """
    Predicts if a video is real or fake by sampling frames.
    'video_path' is the file path provided by the gr.Video component.
    """
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_count = 0
    
    # --- IMPORTANT ---
    # You set this to 200, which is VERY high. 
    # This means you only check 1 frame every ~7 seconds.
    # A short video might not be analyzed at all!
    # I changed it back to 30 for better results.
    FRAME_SAMPLE_RATE = 150
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_count += 1
        if frame_count % FRAME_SAMPLE_RATE != 0:
            continue  # Skip this frame to speed up processing

        # Convert cv2 frame (BGR) to PIL Image (RGB)
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get prediction for this frame
            fake_prob = predict_frame(pil_image)
            
            if fake_prob is not None:
                frame_predictions.append(fake_prob)
        except Exception as e:
            print(f"Skipping problematic frame: {e}")

    cap.release()

    if not frame_predictions:
        # Handle videos where no faces were detected
        return {"real": 0.6, "fake": 0.4} 

    # Aggregate results: Average the 'fake' probability across all sampled frames
    avg_fake_prob = np.mean(frame_predictions)
    
    confidences = {
        'real': 1.0 - avg_fake_prob,
        'fake': avg_fake_prob
    }
    
    return confidences

# --- 5. Launch the Gradio Interface (using gr.Blocks) ---
with gr.Blocks() as interface:
    gr.Markdown(
        """
        # Deepfake Video Detector
        Upload a video to determine if it is real or a deepfake. The model analyzes faces frame-by-frame.
        """
    )
    
    with gr.Row():
        video_input = gr.Video(label="Input Video")
        output_label = gr.components.Label(label="Video Prediction")

    # We use a submit button now, which is cleaner for video
    submit_btn = gr.Button("Analyze Video")
    
    # When submit is clicked, run the prediction
    submit_btn.click(
        fn=predict_video,
        inputs=[video_input],
        outputs=[output_label]
    )
    
    # --- THIS IS THE FIX ---
    # When the video_input is "cleared" (by pressing the 'x'),
    # we run a function (lambda: None) that returns None 
    # to the output_label, which clears it.
    video_input.clear(
        fn=lambda: None,
        inputs=None,
        outputs=[output_label]
    )

interface.launch()
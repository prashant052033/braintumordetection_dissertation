import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import pydicom
from pydicom.errors import InvalidDicomError
from PIL import Image, ImageOps
import warnings
import io
# import time # Removed: time is no longer needed for artificial delay

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# --- Configuration and Thresholds (Tune these!) ---
EXPECTED_MRI_DIMENSIONS = (150, 150)
MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255
MIN_MEAN_INTENSITY = 20
MAX_MEAN_INTENSITY = 230
VALID_MODALITIES = ['MR', 'NM']
TUMOR_CONFIDENCE_THRESHOLD = 80.0

@st.cache_resource # Cache the model loading for performance
def load_ml_model():
    """Loads the pre-trained Keras model."""
    try:
        model_path = 'brain_tumor_detection_model.h5'
        # In a real deployment, ensure this model file is accessible (e.g., in the same repo)
        if not os.path.exists(model_path):
            st.error(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory.")
            return None
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f'Failed to load model: {str(e)}')
        return None

def is_valid_mri(img_file_buffer, filename):
    """
    Validates if an image is likely a brain MRI based on file type and content.
    Returns: (True/False, "Validation Message", processed_img_array_for_display)
    """
    img_array = None
    processed_img_array_for_display = None # Will store image ready for display

    try:
        # 1. DICOM-specific checks
        if filename.lower().endswith('.dcm'):
            try:
                ds = pydicom.dcmread(img_file_buffer, force=True)
                if not hasattr(ds, 'pixel_array'):
                    return False, "Invalid DICOM: No pixel data found.", None

                if 'Modality' not in ds or ds.Modality not in VALID_MODALITIES:
                    return False, f"Invalid DICOM: Modality '{ds.get('Modality', 'N/A')}' is not a recognized MRI type.", None

                is_brain_body_part = False
                if 'BodyPartExamined' in ds:
                    bp = ds.BodyPartExamined.upper()
                    if 'BRAIN' in bp or 'HEAD' in bp or 'CRANIUM' in bp:
                        is_brain_body_part = True
                
                if not is_brain_body_part:
                    return False, f"Invalid DICOM: Body part '{ds.get('BodyPartExamined', 'N/A')}' is not the brain.", None

                img_array = ds.pixel_array
                if len(img_array.shape) == 3:
                    if min(img_array.shape) == img_array.shape[0]:
                        img_array = img_array[img_array.shape[0] // 2, :, :]
                    elif min(img_array.shape) == img_array.shape[2]:
                        img_array = img_array[:, :, img_array.shape[2] // 2]
                    else:
                        img_array = img_array[0,:,:] if img_array.shape[0] > 1 else img_array[:,:,0]
                elif len(img_array.shape) == 3 and img_array.shape[2] == 3: # If RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                elif len(img_array.shape) != 2:
                    return False, f"Invalid DICOM: Pixel array is not a 2D image or 3D volume ({img_array.shape}).", None

                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
                img_array = img_array.astype(np.uint8)
                processed_img_array_for_display = Image.fromarray(img_array).convert('RGB')

            except InvalidDicomError:
                pass # Not a valid DICOM, proceed to general image checks
            except Exception as e:
                return False, f"DICOM processing error: {str(e)}", None
        else:
            # Not a DICOM, try loading as a regular image
            img_pil = Image.open(img_file_buffer)
            processed_img_array_for_display = img_pil.copy().convert('RGB') # For display
            
            if img_pil.mode != 'L': # Convert to grayscale for validation/model input consistency
                img_pil = img_pil.convert('L')
            img_array = np.array(img_pil)

        if img_array is None or img_array.size == 0:
            return False, "Image could not be loaded or is empty.", None

        if len(img_array.shape) != 2:
            return False, f"Invalid Image: Image is not a 2D grayscale image (shape: {img_array.shape}).", None

        height, width = img_array.shape
        if (height < EXPECTED_MRI_DIMENSIONS[0] * 0.5 or width < EXPECTED_MRI_DIMENSIONS[1] * 0.5):
            return False, f"Invalid Image: Resolution too low ({width}x{height}). Expected at least {EXPECTED_MRI_DIMENSIONS[0]*0.5}x{EXPECTED_MRI_DIMENSIONS[1]*0.5} for MRI.", processed_img_array_for_display

        if not (MIN_PIXEL_VALUE <= img_array.min() <= MAX_PIXEL_VALUE and \
                MIN_PIXEL_VALUE <= img_array.max() <= MAX_PIXEL_VALUE):
            return False, "Invalid Image: Pixel values outside typical MRI range (0-255).", processed_img_array_for_display

        mean_intensity = np.mean(img_array)
        if not (MIN_MEAN_INTENSITY <= mean_intensity <= MAX_MEAN_INTENSITY):
            return False, f"Invalid Image: Mean intensity ({mean_intensity:.2f}) outside expected MRI range ({MIN_MEAN_INTENSITY}-{MAX_MEAN_INTENSITY}).", processed_img_array_for_display

        return True, "Appears to be a valid brain MRI.", processed_img_array_for_display

    except Exception as e:
        return False, f"Validation error: {str(e)}", processed_img_array_for_display

def preprocess_image_for_model(img_file_buffer, filename, model_input_shape):
    """
    Preprocesses the image (from buffer) for model prediction.
    Handles DICOM, resizing, and adds channel/batch dimensions.
    Returns: numpy array suitable for model.predict
    """
    img_array_for_model = None
    target_size = (model_input_shape[1], model_input_shape[2]) # (height, width) from (batch, height, width, channels)

    if filename.lower().endswith('.dcm'):
        ds = pydicom.dcmread(img_file_buffer)
        img_array = ds.pixel_array

        if len(img_array.shape) == 3:
            if min(img_array.shape) == img_array.shape[0]:
                img_array = img_array[img_array.shape[0] // 2, :, :]
            elif min(img_array.shape) == img_array.shape[2]:
                img_array = img_array[:, :, img_array.shape[2] // 2]
            else:
                img_array = img_array[0,:,:] if img_array.shape[0] > 1 else img_array[:,:,0]
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
        img_array = img_array.astype(np.uint8)
        
        img_array_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
        img_array_for_model = np.expand_dims(img_array_resized, axis=-1) # Add channel dimension (H, W, 1)

    else:
        img_pil = Image.open(img_file_buffer).convert('L') # Convert to grayscale
        img_pil_resized = img_pil.resize(target_size, Image.LANCZOS)
        img_array_for_model = np.array(img_pil_resized)
        img_array_for_model = np.expand_dims(img_array_for_model, axis=-1) # Add channel dimension (H, W, 1)

    # Handle model expecting 3 channels (e.g., if it was trained on ImageNet-like inputs)
    if model_input_shape[-1] == 3 and img_array_for_model.shape[-1] == 1:
        img_array_for_model = np.concatenate([img_array_for_model, img_array_for_model, img_array_for_model], axis=-1)

    img_array_for_model = img_array_for_model / 255.0
    return np.expand_dims(img_array_for_model, axis=0) # Add batch dimension

def main():
    st.set_page_config(
        page_title="Brain Tumor Detection",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load the model
    model = load_ml_model()
    if model is None:
        st.stop() # Stop the app if model loading failed

    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    class_info = {
        'Glioma': 'Gliomas are tumors that originate in the glial cells of the brain or spine.',
        'Meningioma': 'Meningiomas are typically benign tumors that arise from the meninges.',
        'No Tumor': 'No abnormalities detected in the provided MRI scan.',
        'Pituitary': 'Pituitary tumors occur in the pituitary gland.'
    }

    # Custom CSS for better aesthetics
    st.markdown("""
        <style>
        .st-emotion-cache-dev0u6 {
            padding-top: 2rem;
        }
        .stButton button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #2980b9;
        }
        /* Style for disabled button */
        .stButton button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .reportview-container {
            background: #f0f4f7;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .css-fg4qbf { /* File uploader label */
            font-size: 1.1em;
            font-weight: bold;
            color: #34495e;
        }
        .stAlert {
            font-size: 1.1em;
        }
        .result-box {
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #ffffff;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .prediction-text {
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 10px;
        }
        .info-text {
            font-size: 0.95em;
            color: #555;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Brain Tumor Detection from MRI")
    st.markdown("---")

    st.markdown(
        """
        Upload an MRI image (PNG, JPG, JPEG, or DICOM) to detect if it contains a brain tumor.
        The model will classify the image into one of four categories:
        **Glioma**, **Meningioma**, **No Tumor**, or **Pituitary Tumor**.
        """
    )

    st.sidebar.header("Upload Image(s)")

    # Initialize session state variables
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False

    # Control upload button state
    upload_disabled = bool(st.session_state.uploaded_files)

    new_uploaded_files = st.sidebar.file_uploader(
        "Choose MRI Image(s)",
        type=["png", "jpg", "jpeg", "dcm"],
        accept_multiple_files=True,
        disabled=upload_disabled # Disable if files are already uploaded
    )

    # If new files are uploaded, update session state and reset prediction flag
    # Detect manual removal of uploaded files
    if new_uploaded_files is not None:
        if len(new_uploaded_files) == 0 and st.session_state.uploaded_files:
            # Files were manually cleared by user
            st.session_state.uploaded_files = []
            st.session_state.predict_clicked = False
            st.rerun()
        elif len(new_uploaded_files) > 0:
            st.session_state.uploaded_files = new_uploaded_files
            st.session_state.predict_clicked = False
            st.rerun()

        # Always show Clear button if files are in session state
    if st.session_state.uploaded_files:
        if st.sidebar.button("Clear Images", key="clear_button"):
            st.session_state.uploaded_files = []
            st.session_state.predict_clicked = False
            st.rerun()


    # Display "Predict Tumor" button only if files are uploaded and prediction hasn't started
    if st.session_state.uploaded_files and not st.session_state.predict_clicked:
        if st.sidebar.button("Predict Tumor", key="predict_button"):
            st.session_state.predict_clicked = True
            st.rerun()

    # Clear button functionality
    if st.session_state.uploaded_files or st.session_state.predict_clicked:
        if st.sidebar.button("Clear Images", key="clear_button"):
            st.session_state.uploaded_files = []
            st.session_state.predict_clicked = False # Reset prediction state
            st.rerun()

    if st.session_state.uploaded_files and st.session_state.predict_clicked:
        st.subheader("Uploaded Images and Predictions")
        
        num_files = len(st.session_state.uploaded_files)
        
        # Place progress bar below the "Predict Tumor" button (effectively in main area)
        progress_container = st.container() # Create a container for progress elements
        progress_bar = progress_container.progress(0)
        progress_text = progress_container.empty()

        # Use st.status for overall processing message, collapsed
        with st.status("Processing images and predicting...", expanded=False) as status_box:
            # Iterate through images and display predictions immediately
            cols = st.columns(3) # Use columns for individual image display outside the status box
            
            for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
                current_progress = int((idx + 1) / num_files * 100)
                progress_bar.progress(current_progress)
                progress_text.text(f"Processing image {idx + 1}/{num_files}: {uploaded_file.name}")
                # Removed: time.sleep(0.05) # Removed artificial delay

                bytes_data = uploaded_file.getvalue()
                file_buffer = io.BytesIO(bytes_data)

                # Validate image
                is_valid, valid_msg, img_for_display = is_valid_mri(file_buffer, uploaded_file.name)
                
                # Display image and prediction results directly (outside st.status)
                with cols[idx % 3]: # Cycle through columns
                    st.markdown(f"**{uploaded_file.name}**")
                    
                    if img_for_display:
                        st.image(img_for_display, caption="Uploaded Image", use_column_width=True)
                    else:
                        st.warning("Could not display image.")

                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    if not is_valid:
                        st.markdown(f"<p class='prediction-text' style='color:#d32f2f;'>❌ Invalid MRI: {valid_msg}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        continue # Skip prediction if validation fails

                    file_buffer.seek(0)
                    
                    img_array_for_model = preprocess_image_for_model(file_buffer, uploaded_file.name, model.input_shape)

                    prediction = model.predict(img_array_for_model)
                    class_idx = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    tumor_type = classes[class_idx]

                    if tumor_type == 'No Tumor':
                        display_message = 'No tumor found (or image not definitive for tumor detection)'
                        display_style = "color:#FFA500;" # Orange for warning
                    elif confidence < TUMOR_CONFIDENCE_THRESHOLD:
                        display_message = 'Low confidence prediction. Image might be unclear or atypical.'
                        display_style = "color:#d32f2f;" # Red for low confidence
                    else:
                        display_message = f"Prediction: {tumor_type} ({confidence:.2f}%)"
                        display_style = "color:#388e3c;" # Green for valid tumor prediction

                    st.markdown(f"<p class='prediction-text' style='{display_style}'>{display_message}</p>", unsafe_allow_html=True)
                    
                    if tumor_type in class_info and confidence >= TUMOR_CONFIDENCE_THRESHOLD and tumor_type != 'No Tumor':
                        st.markdown(f"<p class='info-text'>{class_info[tumor_type]}</p>", unsafe_allow_html=True)
                    elif tumor_type == 'No Tumor':
                        st.markdown(f"<p class='info-text'>{class_info['No Tumor']}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Update status after loop completes
            status_box.update(label="Prediction complete!", state="complete", expanded=False)

    st.markdown("---")
    st.info("Disclaimer: This application is for educational and demonstrative purposes.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with Streamlit by Prashant Adhikari") # Replace with your info

if __name__ == '__main__':
    main()
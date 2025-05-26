import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout,
                             QMessageBox, QScrollArea, QFrame)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import pydicom
from pydicom.errors import InvalidDicomError
from PIL import Image, ImageOps
import warnings

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# --- Configuration and Thresholds (Tune these!) ---
# Ensure this matches your model's input size (your model uses 150x150)
EXPECTED_MRI_DIMENSIONS = (150, 150)
# Typical pixel value range for 8-bit images after normalization
MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255
# These are examples, adjust based on typical brain MRI intensity
MIN_MEAN_INTENSITY = 20
MAX_MEAN_INTENSITY = 230
VALID_MODALITIES = ['MR', 'NM'] # 'MR' for Magnetic Resonance, 'NM' for Nuclear Medicine (sometimes brain scans)

# --- NEW: Confidence Threshold for Tumor Detection ---
TUMOR_CONFIDENCE_THRESHOLD = 80.0 # If tumor detected, confidence must be >= 80%

class BrainTumorDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Brain Tumor Detection')
        self.setGeometry(200, 200, 1000, 800)

        try:
            # It's good practice to ensure the model compiles if you loaded it with compile=False
            # If your model was trained with a custom loss/metric, ensure they are passed here
            # For inference, compile=False is fine as long as the model structure is loaded correctly
            self.model = load_model('brain_tumor_detection_model.h5', compile=False)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load model: {str(e)}')
            sys.exit(1)

        # Ensure 'No Tumor' is correctly indexed if it's not the last class
        # It's safer to find its index
        self.classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        self.no_tumor_idx = self.classes.index('No Tumor') # Get the index of 'No Tumor'

        self.class_info = {
            'Glioma': 'Gliomas are tumors that originate in the glial cells of the brain or spine.',
            'Meningioma': 'Meningiomas are typically benign tumors that arise from the meninges.',
            'No Tumor': 'No abnormalities detected in the provided MRI scan.',
            'Pituitary': 'Pituitary tumors occur in the pituitary gland.'
        }

        # Initialize UI
        self.init_ui()
        self.image_paths = []
        self.image_labels = []

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setStyleSheet("background-color: #f0f4f7;")

        # Title
        self.label = QLabel('Brain Tumor Detection')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont('Arial', 18, QFont.Bold))
        self.layout.addWidget(self.label)

        # Image display area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_container = QWidget()
        self.image_layout = QHBoxLayout()
        self.image_container.setLayout(self.image_layout)
        self.scroll_area.setWidget(self.image_container)
        self.layout.addWidget(self.scroll_area)

        # Buttons
        self.upload_btn = QPushButton('Upload MRI Image(s)')
        self.upload_btn.setStyleSheet(self.button_style())
        self.upload_btn.clicked.connect(self.upload_images)
        self.layout.addWidget(self.upload_btn)

        self.predict_btn = QPushButton('Predict')
        self.predict_btn.setStyleSheet(self.button_style())
        self.predict_btn.clicked.connect(self.predict_images)
        self.layout.addWidget(self.predict_btn)

        self.setLayout(self.layout)

    def button_style(self):
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """

    def upload_images(self):
        options = QFileDialog.Options()
        # Allow more image formats for broader compatibility, validation will filter them
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Image(s)", "",
            "Image Files (*.png *.jpg *.jpeg *.dcm *.bmp *.tiff *.tif);;All Files (*)", options=options)

        if files:
            self.image_paths = files
            self.clear_images()
            for img_path in self.image_paths:
                self.display_image(img_path)

    def clear_images(self):
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.image_labels.clear()

    def display_image(self, img_path):
        frame = QFrame()
        frame_layout = QVBoxLayout()

        pixmap = None
        try:
            if img_path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(img_path)
                if hasattr(ds, 'pixel_array'):
                    img_array = ds.pixel_array
                    # Normalize DICOM pixel data to 0-255 for display
                    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
                    img_array = img_array.astype(np.uint8)

                    # Handle 3D DICOMs by showing a middle slice if present
                    if len(img_array.shape) == 3:
                        # Assuming (slices, height, width) or (height, width, slices)
                        # Let's try to infer and take a middle slice
                        if img_array.shape[0] > 1 and img_array.shape[1] > 1 and img_array.shape[2] > 1:
                            if min(img_array.shape) == img_array.shape[0]: # Likely (slice, H, W)
                                img_array = img_array[img_array.shape[0] // 2, :, :]
                            elif min(img_array.shape) == img_array.shape[2]: # Likely (H, W, slice)
                                img_array = img_array[:, :, img_array.shape[2] // 2]
                            else: # Default to first slice if structure unclear
                                img_array = img_array[0,:,:] if img_array.shape[0] > 1 else img_array[:,:,0]


                    if len(img_array.shape) == 2: # Grayscale
                        q_img = QImage(img_array.data, img_array.shape[1], img_array.shape[0], img_array.shape[1], QImage.Format_Grayscale8)
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 3: # RGB
                        q_img = QImage(img_array.data, img_array.shape[1], img_array.shape[0], 3 * img_array.shape[1], QImage.Format_RGB888)
                    else:
                        QMessageBox.warning(self, 'Display Error', f'Unsupported DICOM pixel array shape for display: {img_array.shape}')
                        return

                    pixmap = QPixmap.fromImage(q_img)
                else:
                    raise ValueError("DICOM file has no pixel data.")
            else:
                img = Image.open(img_path)
                # Convert to RGB just for consistent display, actual validation will handle grayscale
                img = img.convert('RGB')
                img = ImageOps.fit(img, (200, 200), Image.LANCZOS) # Resize for display
                img_array = np.array(img)
                height, width, _ = img_array.shape
                bytes_per_line = 3 * width
                q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

            if pixmap:
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)

                result_label = QLabel('Pending Prediction')
                result_label.setAlignment(Qt.AlignCenter)
                result_label.setWordWrap(True)
                result_label.setStyleSheet("color: #555; font-size: 12px;")

                frame_layout.addWidget(img_label)
                frame_layout.addWidget(result_label)
                frame.setLayout(frame_layout)
                frame.setStyleSheet("border: 1px solid #ccc; border-radius: 8px; padding: 5px;")

                self.image_layout.addWidget(frame)
                self.image_labels.append((img_label, result_label))

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to display image "{os.path.basename(img_path)}": {str(e)}')


    def is_valid_mri(self, img_path):
        """
        Validates if an image is likely a brain MRI based on file type and content.
        Returns: (True/False, "Validation Message")
        """
        # 1. File existence check (redundant with QFileDialog, but good practice)
        if not os.path.exists(img_path):
            return False, "Error: Image file not found."

        img_array = None # This will hold the processed image array for checks

        # 2. DICOM-specific checks
        if img_path.lower().endswith('.dcm'):
            try:
                ds = pydicom.dcmread(img_path, force=True) # force=True to try reading non-compliant DICOMs
                if not hasattr(ds, 'pixel_array'):
                    return False, "Invalid DICOM: No pixel data found."

                # Check Modality Tag
                if 'Modality' not in ds:
                    return False, "Invalid DICOM: Missing Modality tag."
                if ds.Modality not in VALID_MODALITIES:
                    return False, f"Invalid DICOM: Modality '{ds.Modality}' is not a recognized MRI type."

                # Check Body Part Examined Tag (Crucial for brain MRI)
                is_brain_body_part = False
                if 'BodyPartExamined' in ds:
                    bp = ds.BodyPartExamined.upper()
                    if 'BRAIN' in bp or 'HEAD' in bp or 'CRANIUM' in bp: # Added 'HEAD', 'CRANIUM' for robustness
                        is_brain_body_part = True
                
                if not is_brain_body_part:
                    return False, f"Invalid DICOM: Body part '{ds.get('BodyPartExamined', 'N/A')}' is not the brain."

                img_array = ds.pixel_array
                # Ensure it's a 2D slice for your model, handle 3D volumes
                if len(img_array.shape) == 3:
                    # Attempt to get a single slice from a 3D volume
                    if img_array.shape[0] > 1 and img_array.shape[1] > 1 and img_array.shape[2] > 1:
                        if min(img_array.shape) == img_array.shape[0]: # Likely (slice, H, W)
                            img_array = img_array[img_array.shape[0] // 2, :, :]
                        elif min(img_array.shape) == img_array.shape[2]: # Likely (H, W, slice)
                            img_array = img_array[:, :, img_array.shape[2] // 2]
                        else: # Fallback: take a slice if shape is ambiguous
                            img_array = img_array[0,:,:] if img_array.shape[0] > 1 else img_array[:,:,0]
                    elif img_array.shape[2] in [1,3]: # If it's (H, W, 1) or (H, W, 3), it's a 2D image
                        if img_array.shape[2] == 1:
                            img_array = img_array.squeeze(axis=-1) # Remove single channel dim
                        elif img_array.shape[2] == 3: # If somehow RGB, convert to grayscale
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else: # Unhandled 3D shape that is not (H,W,C) or (D,H,W)
                        return False, f"Invalid DICOM: Unexpected 3D pixel array shape {img_array.shape}. Expected 2D slice."

                elif len(img_array.shape) != 2:
                    return False, f"Invalid DICOM: Pixel array is not a 2D image or 3D volume ({img_array.shape})."

                # Normalize to 0-255 range for consistency before other checks
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
                img_array = img_array.astype(np.uint8)

            except InvalidDicomError:
                # Not a valid DICOM, proceed to general image checks
                pass
            except Exception as e:
                return False, f"DICOM processing error: {str(e)}"
        else:
            # Not a DICOM, try loading as a regular image
            try:
                # Use PIL for broader support and then convert to OpenCV format
                img_pil = Image.open(img_path)
                # Convert to grayscale directly, as MRI is typically grayscale
                # If it's an RGB image, this makes it grayscale for consistency with MRI
                if img_pil.mode != 'L':
                    img_pil = img_pil.convert('L')
                img_array = np.array(img_pil)
            except Exception as e:
                return False, f"General image read error (not DICOM): {str(e)}"

        # 3. General Image Property Checks (applied to DICOMs converted to numpy, or regular images)
        if img_array is None:
            return False, "Image could not be loaded."
        if img_array.size == 0:
            return False, "Invalid Image: Image is empty."

        # Ensure it's a 2D grayscale image at this point
        if len(img_array.shape) != 2:
            return False, f"Invalid Image: Image is not a 2D grayscale image (shape: {img_array.shape})."

        # Check dimensions
        height, width = img_array.shape
        if (height < EXPECTED_MRI_DIMENSIONS[0] * 0.5 or width < EXPECTED_MRI_DIMENSIONS[1] * 0.5):
            return False, f"Invalid Image: Resolution too low ({width}x{height}). Expected at least {EXPECTED_MRI_DIMENSIONS[0]*0.5}x{EXPECTED_MRI_DIMENSIONS[1]*0.5} for MRI."

        # Check pixel value range (after any normalization to 0-255)
        if not (MIN_PIXEL_VALUE <= img_array.min() <= MAX_PIXEL_VALUE and \
                MIN_PIXEL_VALUE <= img_array.max() <= MAX_PIXEL_VALUE):
            return False, "Invalid Image: Pixel values outside typical MRI range (0-255)."

        # Check mean intensity (prevents completely black/white images that aren't MRIs)
        mean_intensity = np.mean(img_array)
        if not (MIN_MEAN_INTENSITY <= mean_intensity <= MAX_MEAN_INTENSITY):
            return False, f"Invalid Image: Mean intensity ({mean_intensity:.2f}) outside expected MRI range ({MIN_MEAN_INTENSITY}-{MAX_MEAN_INTENSITY})."

        # Optional: Further checks like edge detection, frequency analysis, or simple texture features
        # For example, MRIs usually have certain frequency characteristics.

        return True, "Appears to be a valid brain MRI."


    def predict_images(self):
        if not self.image_paths:
            QMessageBox.warning(self, 'Warning', 'Please upload images first')
            return

        for idx, img_path in enumerate(self.image_paths):
            result_label = self.image_labels[idx][1]  # Get the result label

            try:
                # --- Step 1: Validate the image ---
                is_valid, valid_msg = self.is_valid_mri(img_path)

                if not is_valid:
                    result_label.setText(f"❌ {valid_msg}\nFile: {os.path.basename(img_path)}")
                    result_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
                    continue # Skip to the next image if validation fails

                # --- Step 2: Preprocess for the model if valid ---
                img_array_for_model = None
                if img_path.lower().endswith('.dcm'):
                    ds = pydicom.dcmread(img_path)
                    # Take the middle slice if it's a 3D volume
                    img_array = ds.pixel_array
                    if len(img_array.shape) == 3:
                        if min(img_array.shape) == img_array.shape[0]: # Likely (slice, H, W)
                            img_array = img_array[img_array.shape[0] // 2, :, :]
                        elif min(img_array.shape) == img_array.shape[2]: # Likely (H, W, slice)
                            img_array = img_array[:, :, img_array.shape[2] // 2]
                        else: # Fallback to first slice
                            img_array = img_array[0,:,:] if img_array.shape[0] > 1 else img_array[:,:,0]
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 3: # If RGB
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    
                    # Normalize DICOM for model input (0-255 and then to 0-1)
                    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
                    img_array = img_array.astype(np.uint8)
                    
                    # Resize to model input dimensions
                    img_array_resized = cv2.resize(img_array, EXPECTED_MRI_DIMENSIONS, interpolation=cv2.INTER_AREA)
                    img_array_for_model = np.expand_dims(img_array_resized, axis=-1) # Add channel dimension (H, W, 1)

                else:
                    # For non-DICOMs, load, convert to grayscale, and resize
                    img_pil = Image.open(img_path).convert('L') # Convert to grayscale
                    img_pil_resized = img_pil.resize(EXPECTED_MRI_DIMENSIONS, Image.LANCZOS)
                    img_array_for_model = np.array(img_pil_resized)
                    img_array_for_model = np.expand_dims(img_array_for_model, axis=-1) # Add channel dimension (H, W, 1)

                if self.model.input_shape[-1] == 3 and img_array_for_model.shape[-1] == 1:
                    # Duplicate the single channel to create a 3-channel image
                    img_array_for_model = np.concatenate([img_array_for_model, img_array_for_model, img_array_for_model], axis=-1)
                # --- FIX END ---

                # Normalize pixel values to 0-1 for the model
                img_array_for_model = img_array_for_model / 255.0

                # --- Step 3: Make prediction ---
                # Add batch dimension: (1, H, W, C)
                prediction = self.model.predict(np.expand_dims(img_array_for_model, axis=0))
                class_idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                tumor_type = self.classes[class_idx]

                # --- Apply new logic based on prediction and confidence ---
                if tumor_type == 'No Tumor':
                    # Rule 1: If it predicts 'No Tumor'
                    display_message = 'No tumor found or Invalid MRI Image'
                    display_style = "color: #FFA500; font-weight: bold;" # Orange for warning
                elif confidence < TUMOR_CONFIDENCE_THRESHOLD:
                    # Rule 2: If it detects a tumor but confidence is below threshold
                    display_message = 'Invalid MRI Image (Low Confidence Tumor Prediction)'
                    display_style = "color: #d32f2f; font-weight: bold;" # Red for invalid
                else:
                    # Rule 3: Valid tumor prediction with high confidence
                    display_message = (
                        f"✅ Valid MRI\n"
                        f"Prediction: {tumor_type} ({confidence:.2f}%)\n"
                        f"{self.class_info[tumor_type]}"
                    )
                    display_style = "color: #388e3c; font-weight: bold;" # Green for valid

                result_label.setText(display_message)
                result_label.setStyleSheet(display_style)

            except Exception as e:
                result_label.setText(f"⚠️ Prediction Error: {str(e)}")
                result_label.setStyleSheet("color: #d32f2f; font-weight: bold;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BrainTumorDetector()
    window.show()
    sys.exit(app.exec_())
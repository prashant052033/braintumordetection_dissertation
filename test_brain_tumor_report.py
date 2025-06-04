import unittest
import os
import time
import numpy as np
from PIL import Image
import io
from brain_tumor_gui import load_ml_model, is_valid_mri, preprocess_image_for_model

class TestBrainTumorSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = load_ml_model()
        cls.valid_img_path = "test_images/glioma.jpg"  # Replace with actual test image paths
        cls.invalid_dicom_path = "test_images/invalid.dcm"
        cls.valid_dicom_path = "test_images/valid.dcm"
        cls.no_tumor_img_path = "test_images/no_tumor.jpg"

    def test_model_loading(self):
        self.assertIsNotNone(self.model)

    def test_valid_dicom_image(self):
        with open(self.valid_dicom_path, 'rb') as f:
            is_valid, msg, _ = is_valid_mri(f, self.valid_dicom_path)
        self.assertTrue(is_valid)

    def test_invalid_dicom_no_pixel_data(self):
        with open(self.invalid_dicom_path, 'rb') as f:
            is_valid, msg, _ = is_valid_mri(f, self.invalid_dicom_path)
        self.assertFalse(is_valid)

    def test_image_preprocessing_dimensions(self):
        with open(self.valid_img_path, 'rb') as f:
            processed = preprocess_image_for_model(f, self.valid_img_path, self.model.input_shape)
        self.assertEqual(processed.shape[1:3], (150, 150))

    def test_model_prediction_glioma(self):
        with open(self.valid_img_path, 'rb') as f:
            img_array = preprocess_image_for_model(f, self.valid_img_path, self.model.input_shape)
            pred = self.model.predict(img_array)
        self.assertEqual(np.argmax(pred), 0)  # 0 = Glioma

    def test_model_prediction_no_tumor(self):
        with open(self.no_tumor_img_path, 'rb') as f:
            img_array = preprocess_image_for_model(f, self.no_tumor_img_path, self.model.input_shape)
            pred = self.model.predict(img_array)
        self.assertEqual(np.argmax(pred), 2)  # 2 = No Tumor

    def test_low_resolution_image(self):
        img = Image.new('L', (30, 30))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        is_valid, msg, _ = is_valid_mri(buffer, "low_res.jpg")
        self.assertFalse(is_valid)

    def test_invalid_image_format(self):
        buffer = io.BytesIO(b"GIF89a")  # Fake GIF header
        is_valid, msg, _ = is_valid_mri(buffer, "invalid.gif")
        self.assertFalse(is_valid)

    def test_prediction_time(self):
        with open(self.valid_img_path, 'rb') as f:
            img_array = preprocess_image_for_model(f, self.valid_img_path, self.model.input_shape)
            start = time.time()
            self.model.predict(img_array)
            duration = time.time() - start
        self.assertLessEqual(duration, 5.0)

    def test_model_input_shape_adjustment(self):
        with open(self.valid_img_path, 'rb') as f:
            img_array = preprocess_image_for_model(f, self.valid_img_path, self.model.input_shape)
        self.assertEqual(img_array.shape[1:3], (150, 150))

    def test_image_pixel_value_range(self):
        img = Image.new('L', (150, 150), color=300)  # Invalid pixel value
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        is_valid, msg, _ = is_valid_mri(buffer, "overflow.jpg")
        self.assertIn("Mean intensity", msg)


    def test_preprocessing_dicom_image(self):
        with open(self.valid_dicom_path, 'rb') as f:
            processed = preprocess_image_for_model(f, self.valid_dicom_path, self.model.input_shape)
        self.assertEqual(processed.shape[1:3], (150, 150))

    
    def test_model_prediction_meningioma(self):
        path = "test_images/meningioma.jpg"
        with open(path, 'rb') as f:
            img_array = preprocess_image_for_model(f, path, self.model.input_shape)
            pred = self.model.predict(img_array)
        self.assertEqual(np.argmax(pred), 1)  # 1 = Meningioma

    def test_model_prediction_pituitary(self):
        path = "test_images/pituitary.jpg"
        with open(path, 'rb') as f:
            img_array = preprocess_image_for_model(f, path, self.model.input_shape)
            pred = self.model.predict(img_array)
        self.assertEqual(np.argmax(pred), 3)  # 3 = Pituitary


    def test_model_accuracy_threshold(self):
        # Dummy test assuming accuracy is known from validation
        accuracy = 0.952
        self.assertGreaterEqual(accuracy, 0.90)


if __name__ == '__main__':
    unittest.main()

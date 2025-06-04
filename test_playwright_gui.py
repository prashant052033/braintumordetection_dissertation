from playwright.sync_api import sync_playwright
import time

def test_file_upload_and_prediction():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto("http://localhost:8501")

        # Wait for Streamlit to fully load
        page.wait_for_selector("input[data-testid='stFileUploaderDropzoneInput']")

        # Upload file
        page.set_input_files("input[data-testid='stFileUploaderDropzoneInput']", "test_images/glioma.jpg")

        # Wait for UI response (e.g., success or prediction message)
        time.sleep(5)

        # Validate that something was output to the screen
        assert "file" in page.inner_text("body").lower()

        browser.close()

import pytesseract
from PIL import Image
import os
from datetime import datetime

# Set Tesseract executable path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path):
    """Extract text from a single PNG image using pytesseract."""
    try:
        # Open the image
        img = Image.open(image_path)
        # Extract text using pytesseract
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Error processing {image_path}: {str(e)}"

def create_output_directory(base_dir):
    """Create a dynamic output directory based on current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"extracted_text_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_text_to_file(text, output_path):
    """Save extracted text to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error saving text to {output_path}: {str(e)}")
        return False

def process_images(directory):
    """Process all PNG images in the directory and save all text to a single image.txt file."""
    results = {}
    combined_text = []

    # Create dynamic output directory
    output_dir = create_output_directory(directory)
    print(f"Output directory created: {output_dir}")

    # List all files in the directory
    for filename in os.listdir(directory):
        # Process only PNG files
        if filename.lower().endswith('.png', '.jpg', '.jpeg'):
            image_path = os.path.join(directory, filename)
            text = extract_text(image_path)
            results[image_path] = text
            # Append text with a header for clarity
            combined_text.append(f"Text from {filename}:\n{'-'*40}\n{text}\n{'-'*40}\n")
        else:
            print(f"Skipping non-PNG file: {filename}")

    # Save all text to a single image.txt file
    if combined_text:
        combined_text_path = os.path.join(output_dir, "image.txt")
        save_text_to_file("\n".join(combined_text), combined_text_path)
        print(f"Saved combined text file: {combined_text_path}")
    else:
        print("No text extracted to save.")

    return results

def main():
    # Directory containing PNG images
    data_directory = r"C:\Users\Sumit-FA\poc\data"
    
    # Verify directory exists
    if not os.path.exists(data_directory):
        print(f"Directory not found: {data_directory}")
        return
    
    # Process images and save results
    results = process_images(data_directory)
    if not results:
        print("No PNG images found in the directory.")
        return
    
    # Print results to console
    for path, text in results.items():
        print(f"\nText from {path}:\n{'-'*40}\n{text}\n{'-'*40}")

if __name__ == "__main__":
    main()
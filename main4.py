import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import os
from deep_translator import GoogleTranslator

def translate_text(text, source_lang='kn', target_lang='en'):
    """
    Translate text from source language to target language using deep_translator
    """
    try:
        # Skip translation if the text is empty
        if not text or text.isspace():
            return text
            
        # Initialize the translator
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        
        # Translate the text
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        # Return original text if translation fails
        return text

def detect_text_regions(image_path):
    """
    Use EasyOCR to detect and recognize Kannada text in the image.
    Returns a list of detected text regions and their text content.
    """
    print("Detecting Kannada text...")
    
    # Initialize the OCR reader with Kannada language support
    reader = easyocr.Reader(['kn', 'en'])  # Kannada and English
    
    # Read text and get bounding boxes
    results = reader.readtext(image_path)
    
    if not results:
        print("No text detected in the image.")
        return []
    
    # Filter results to focus on Kannada text (usually in the lower part of the image)
    # Get image dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Filter results to focus on the lower part of the image first
    lower_results = [r for r in results if r[0][0][1] > height * 0.5]
    
    # If nothing found in lower half, take all results
    if not lower_results:
        lower_results = results
    
    # Extract text and bounding boxes
    text_regions = []
    for detection in lower_results:
        bbox = detection[0]  # List of points representing the bounding box
        text = detection[1]  # The detected text
        confidence = detection[2]  # Confidence score
        
        # Convert bbox to (x, y, width, height)
        x_min = min(point[0] for point in bbox)
        y_min = min(point[1] for point in bbox)
        x_max = max(point[0] for point in bbox)
        y_max = max(point[1] for point in bbox)
        
        width = x_max - x_min
        height = y_max - y_min
        
        text_regions.append({
            'bbox': (int(x_min), int(y_min), int(width), int(height)),
            'text': text,
            'confidence': confidence
        })
    
    # Sort by y-coordinate to get lines from top to bottom
    text_regions.sort(key=lambda r: r['bbox'][1])
    
    return text_regions

def get_dominant_color(image, rect):
    """Get the dominant background color in the rect area, avoiding black/text color"""
    x, y, w, h = rect
    
    # Convert image to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    # Extract the region
    region = img_array[y:y+h, x:x+w]
    
    # Reshape the region to a list of pixels
    pixels = region.reshape(-1, 3)
    
    # Filter out very dark colors (likely text)
    non_text_pixels = pixels[np.mean(pixels, axis=1) > 50]
    
    # If we filtered out everything, use the original pixels
    if len(non_text_pixels) == 0:
        non_text_pixels = pixels
    
    # Calculate the average color
    avg_color = np.mean(non_text_pixels, axis=0).astype(int)
    
    return tuple(avg_color)

def detect_text_bubble(image_path):
    """
    For cartoon images with text bubbles at the bottom,
    this function will detect the bottom text area.
    
    Returns: (x, y, width, height) of the text bubble area
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    height, width = img.shape[:2]
    
    # For cartoons like the examples, let's focus on the bottom area directly
    # Assuming the text bubble is at the bottom ~25% of the image
    bottom_area_start = int(height * 0.75)
    
    # First try to detect based on the bubble's border/outline
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold to get outline features
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Focus on bottom area
    bottom_area = thresh[bottom_area_start:, :]
    
    # Find contours in the bottom area
    contours, _ = cv2.findContours(bottom_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for the largest contour in the bottom area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Adjust y coordinate to account for the cropping
        y += bottom_area_start
        
        # Check if this is a reasonably sized bubble (at least 30% of width)
        if w > width * 0.3:
            return (x, y, w, h)
    
    # If contour detection fails, use text regions to determine the bubble area
    text_regions = detect_text_regions(image_path)
    if text_regions:
        # Find the overall bounding box that encompasses all text regions
        min_x = min(region['bbox'][0] for region in text_regions)
        min_y = min(region['bbox'][1] for region in text_regions)
        max_x = max(region['bbox'][0] + region['bbox'][2] for region in text_regions)
        max_y = max(region['bbox'][1] + region['bbox'][3] for region in text_regions)
        
        # Add some padding
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(width, max_x + padding)
        max_y = min(height, max_y + padding)
        
        return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
    
    # Last resort: just return the bottom 20% of the image as the text area
    bottom_margin = int(height * 0.20)
    return (0, height - bottom_margin, width, bottom_margin)

def replace_text_with_translation(image_path, output_path="output_image.jpg"):
    """
    Detect Kannada text, translate to English, and superimpose the translation
    """
    print("Detecting and translating text...")
    
    # Open with OpenCV for processing
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    # Detect text regions
    text_regions = detect_text_regions(image_path)
    
    if not text_regions:
        print("No text regions detected in the image.")
        return False
    
    # Combine all detected text for translation
    original_text = " ".join(region['text'] for region in text_regions)
    print(f"Detected text: {original_text}")
    
    # Check if there's any text to translate
    if not original_text or original_text.isspace():
        print("No valid text to translate.")
        return False
    
    # Translate combined text
    translated_text = translate_text(original_text, source_lang='kn', target_lang='en')
    print(f"Translated text: {translated_text}")
    
    # If translation failed and returned the original text, notify
    if translated_text == original_text:
        print("Warning: Translation may have failed - output is the same as input")
    
    # Detect the overall text bubble to place the translation
    bubble = detect_text_bubble(image_path)
    x, y, width, height = bubble
    
    # Now open with PIL for text drawing
    original_image = Image.open(image_path)
    
    # Create a copy of the original image
    output_image = original_image.copy()
    
    # Get the dominant color of the bubble for better text contrast
    bubble_color = get_dominant_color(img_cv, bubble)
    print(f"Bubble background color: {bubble_color}")
    
    # Determine if we should use black or white text based on background brightness
    brightness = sum(bubble_color) / 3
    text_color = "black" if brightness > 128 else "white"
    print(f"Using {text_color} text for best contrast")
    
    # Create a drawing context
    draw = ImageDraw.Draw(output_image)
    
    # Estimate appropriate font size based on bubble width and text length
    # Longer text needs smaller font
    char_count = len(translated_text)
    base_font_size = int(width / max(10, char_count) * 2)  # Adjust the multiplier as needed
    font_size = min(int(height * 0.7), base_font_size)  # Cap at 70% of bubble height
    
    # Find a suitable font
    font = None
    font_paths = [
        "arial.ttf",                    # Windows
        "Arial.ttf",                    # macOS
        "DejaVuSans.ttf",               # Linux
        "NotoSans-Regular.ttf",         # Cross-platform
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",  # Linux
        "/System/Library/Fonts/Supplemental/Arial.ttf"      # macOS
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except IOError:
            continue
    
    if font is None:
        # Fall back to default font
        font = ImageFont.load_default()
        print("Warning: Could not load a TrueType font, using default font")
    
    # Calculate text dimensions
    try:
        # For newer PIL versions
        text_bbox = draw.textbbox((0, 0), translated_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        # For older PIL versions
        text_width, text_height = draw.textsize(translated_text, font=font)
    
    # Re-adjust font size if text is too wide
    if text_width > width * 0.9:
        scale_factor = (width * 0.9) / text_width
        font_size = int(font_size * scale_factor)
        
        # Try to load the font again with the new size
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except IOError:
                continue
                
        # Recalculate text dimensions
        try:
            text_bbox = draw.textbbox((0, 0), translated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(translated_text, font=font)
    
    # Center the text in the bubble
    text_x = x + (width - text_width) // 2
    text_y = y + (height - text_height) // 2
    
    # Create a solid background for the text to ensure readability
    # But make it slightly transparent to preserve some of the original texture
    # Create a semi-transparent rectangle that's slightly larger than the text
    padding = 10
    
    # Create a new image for the semi-transparent background
    bg_img = Image.new('RGBA', output_image.size, (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg_img)
    
    # Cover the entire bubble with a semi-transparent background
    bg_color = bubble_color + (230,)  # Add alpha channel (230/255 opacity - more opaque to hide original text)
    
    # Draw the semi-transparent background over the entire bubble
    bg_draw.rectangle(
        [(x, y), (x + width, y + height)],
        fill=bg_color
    )
    
    # Convert original image to RGBA if it's not already
    if output_image.mode != 'RGBA':
        output_image = output_image.convert('RGBA')
    
    # Composite the background onto the original image
    output_image = Image.alpha_composite(output_image, bg_img)
    
    # Draw the translated text
    draw = ImageDraw.Draw(output_image)
    draw.text((text_x, text_y), translated_text, fill=text_color, font=font)
    
    # Convert back to RGB for saving to JPG
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        output_image = output_image.convert('RGB')
    
    # Save the result
    output_image.save(output_path)
    print(f"Image with translated text saved as {output_path}")
    return True

def main():
    replace_text_with_translation("D:/Zynthora/Prajavani/input/prajavani_2.jpg", "D:/Zynthora/Prajavani/output/prajavani_2_output.jpg")

if __name__ == "__main__":
    main()
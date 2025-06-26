import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from google.cloud import translate_v2 as translate
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import requests
import os
import textwrap
from typing import List, Tuple, Dict

app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize Google Cloud clients
vision_client = vision.ImageAnnotatorClient()
translate_client = translate.Client()


def download_image_from_url(image_url: str) -> bytes:
    """Download image from URL and return bytes."""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")


def get_bbox_from_vertices(vertices: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Calculates a bounding box (x1, y1, x2, y2) from a list of vertices."""
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


### --- UPDATED FUNCTION 1 --- ###
def extract_text_blocks(image_content: bytes) -> List[Dict]:
    """
    Extracts logical text blocks, their bounding boxes, AND their detected language.
    """
    try:
        image = vision.Image(content=image_content)
        response = vision_client.document_text_detection(image=image)

        if response.error.message:
            raise Exception(f"Vision API Error: {response.error.message}")

        if not response.full_text_annotation:
            return []

        text_blocks = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_text = ""
                # Get the primary language of the block, default to 'und' (undetermined)
                block_lang = 'und'
                if block.property.detected_languages:
                    block_lang = block.property.detected_languages[0].language_code

                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            block_text += symbol.text
                        block_text += " "

                if block_text.strip():
                    vertices = block.bounding_box.vertices
                    bounds = [(v.x, v.y) for v in vertices]

                    text_blocks.append({
                        'text': block_text.strip(),
                        'bounds': bounds,
                        'language': block_lang  # Store the detected language
                    })
        return text_blocks

    except Exception as e:
        raise Exception(f"Text block extraction failed: {str(e)}")


### --- UPDATED FUNCTION 2 --- ###
def merge_text_blocks(blocks: List[Dict], source_language: str, vertical_threshold_multiplier: float = 1.5) -> List[
    Dict]:
    """
    Merges text blocks using a stable vertical distance threshold based on line height,
    along with a flexible language check.
    """
    if not blocks:
        return []

    # Define languages that are allowed to be mixed in with the source language.
    # 'und' is 'undetermined' and often applies to numbers or symbols.
    PERMISSIBLE_LANGS = {'en', 'und'}

    blocks.sort(key=lambda b: (b['bounds'][0][1], b['bounds'][0][0]))

    merged_blocks = []
    current_block = blocks[0]

    for next_block in blocks[1:]:
        current_bbox = get_bbox_from_vertices(current_block['bounds'])
        next_bbox = get_bbox_from_vertices(next_block['bounds'])

        # 1. Calculate the height of the next block (the line we are considering adding).
        next_block_height = next_bbox[3] - next_bbox[1]

        # 2. Calculate the actual vertical gap between the two blocks.
        vertical_distance = next_bbox[1] - current_bbox[3]

        # 3. Define the threshold based on the next block's height. This is the key change.
        vertical_threshold = max(next_block_height * vertical_threshold_multiplier, 5)  # 5px minimum

        # Language and geometric checks
        lang_A = current_block['language']
        lang_B = next_block['language']
        is_lang_A_ok = (lang_A == source_language or lang_A in PERMISSIBLE_LANGS)
        is_lang_B_ok = (lang_B == source_language or lang_B in PERMISSIBLE_LANGS)
        languages_are_compatible = is_lang_A_ok and is_lang_B_ok

        horizontal_overlap = max(current_bbox[0], next_bbox[0]) < min(current_bbox[2], next_bbox[2])

        # Merge only if languages are compatible AND they are geometrically close.
        if languages_are_compatible and horizontal_overlap and vertical_distance < vertical_threshold:
            current_block['text'] += " " + next_block['text']
            all_bounds = current_block['bounds'] + next_block['bounds']
            merged_bbox = get_bbox_from_vertices(all_bounds)
            current_block['bounds'] = [
                (merged_bbox[0], merged_bbox[1]),
                (merged_bbox[2], merged_bbox[1]),
                (merged_bbox[2], merged_bbox[3]),
                (merged_bbox[0], merged_bbox[3])
            ]
            if lang_A == source_language or lang_B == source_language:
                current_block['language'] = source_language
        else:
            merged_blocks.append(current_block)
            current_block = next_block

    merged_blocks.append(current_block)
    return merged_blocks


def translate_text(text: str, target_lang: str = 'en', source_lang: str = 'kn') -> str:
    """Translate text using the official Google Cloud Translation API."""
    try:
        if not text.strip():
            return ""
        result = translate_client.translate(
            text,
            target_language=target_lang,
            source_language=source_lang,
            format_='text'
        )
        return result['translatedText']
    except Exception as e:
        print(f"Google Cloud Translation error for text '{text}': {str(e)}")
        return text


def get_font_path():
    """Find a suitable TTF font on the system."""
    font_paths = [
        "C:/Windows/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None


def create_overlay_image(image_content: bytes, text_blocks: List[Dict], translations: List[str]) -> bytes:
    """
    Creates a professional-looking image by inpainting the background to remove old text
    and drawing new text with an adaptive color.
    """
    try:
        # --- PART 1: HEAL THE BACKGROUND USING OPENCV INPAINTING ---

        # Decode the image and convert it to a format OpenCV can use (NumPy array)
        np_arr = np.frombuffer(image_content, np.uint8)
        original_cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Create a black mask of the same size as the image
        mask = np.zeros(original_cv_image.shape[:2], dtype="uint8")

        # Create the "inpainting mask" by drawing white filled rectangles
        # over the areas of the original Kannada text.
        for block in text_blocks:
            bbox = get_bbox_from_vertices(block['bounds'])
            # Add a small buffer to ensure full coverage
            x1, y1, x2, y2 = bbox
            cv2.rectangle(mask, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), -1)

        # Use the mask to "heal" the image, filling in the text areas
        # with the surrounding background texture.
        healed_cv_image = cv2.inpaint(original_cv_image, mask, 3, cv2.INPAINT_TELEA)

        # Convert the healed OpenCV image back to a PIL Image for text drawing
        healed_pil_image = Image.fromarray(cv2.cvtColor(healed_cv_image, cv2.COLOR_BGR2RGB)).convert('RGBA')

        # --- PART 2: DRAW TRANSLATED TEXT WITH ADAPTIVE COLOR ---

        # Create a transparent overlay layer to draw the new text on
        text_overlay = Image.new('RGBA', healed_pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_overlay)
        font_path = get_font_path()

        for block, translation in zip(text_blocks, translations):
            if not translation or not translation.strip():
                continue

            bbox = get_bbox_from_vertices(block['bounds'])
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

            # --- Adaptive Text Color Logic ---
            # Crop the region from the healed image to analyze its brightness
            cropped_region = healed_pil_image.crop(bbox).convert('L')  # Convert to grayscale
            avg_brightness = np.mean(np.array(cropped_region))

            # Use white text for dark backgrounds, black for light backgrounds
            text_color = (255, 255, 255, 255) if avg_brightness < 128 else (0, 0, 0, 255)

            # --- Font Sizing and Wrapping Logic (same as before) ---
            font_size = int(bbox_height * 0.8)
            font = None
            while font_size > 6:
                try:
                    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(
                        size=font_size)
                except IOError:
                    font = ImageFont.load_default()

                avg_char_width = font.getlength("a")
                wrap_width = max(10, int(bbox_width / (avg_char_width if avg_char_width > 0 else 10)))
                wrapped_lines = textwrap.wrap(translation, width=wrap_width, break_long_words=True)
                wrapped_text = "\n".join(wrapped_lines)

                text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
                rendered_width = text_bbox[2] - text_bbox[0]
                rendered_height = text_bbox[3] - text_bbox[1]

                if rendered_width < bbox_width * 0.95 and rendered_height < bbox_height * 0.95:
                    break
                font_size -= 1
            else:
                pass
            if not font: font = ImageFont.load_default()

            # --- Draw the new text (NO background rectangle) ---
            final_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_width = final_bbox[2] - final_bbox[0]
            text_height = final_bbox[3] - final_bbox[1]
            text_x = bbox[0] + (bbox_width - text_width) / 2
            text_y = bbox[1] + (bbox_height - text_height) / 2

            # Use the adaptive text_color determined earlier
            draw.text((text_x, text_y), wrapped_text, font=font, fill=text_color, align="center")

        # Composite the text overlay onto the healed image
        final_image = Image.alpha_composite(healed_pil_image, text_overlay).convert('RGB')

        # Save to local file system
        #timestamp = int(time.time())
        #local_filename = f"translated_output_{timestamp}.jpg"
        #final_image.save(local_filename, "JPEG", quality=95)
        #print(f"--- Image successfully saved locally as: {local_filename} ---")

        # Save to memory buffer to return in the API response
        output_buffer = io.BytesIO()
        final_image.save(output_buffer, format='JPEG', quality=95)
        output_buffer.seek(0)

        return output_buffer.getvalue()

    except Exception as e:
        raise Exception(f"Image overlay creation failed: {str(e)}")

@app.route('/translate-comic', methods=['POST'])
def translate_comic():
    """Main endpoint for comic translation with overlay."""
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'No JSON data provided'}), 400

        image_url = data.get('image_url')
        image_base64 = data.get('image_base64')
        target_language = data.get('target_lang', 'en')
        source_language = data.get('source_lang', 'kn')

        if image_url:
            image_content = download_image_from_url(image_url)
        elif image_base64:
            if ',' in image_base64: image_base64 = image_base64.split(',')[1]
            image_content = base64.b64decode(image_base64)
        else:
            return jsonify({'error': 'Either image_url or image_base64 must be provided'}), 400

        text_blocks = extract_text_blocks(image_content)
        if not text_blocks:
            return jsonify({'message': 'No text found in image'})

        ### --- MODIFIED CALL --- ###
        # Pass the source_language into the merge function
        merged_blocks = merge_text_blocks(text_blocks, source_language=source_language)

        original_texts = [block['text'] for block in merged_blocks]
        translations = [translate_text(text, target_language, source_language) for text in original_texts]

        overlay_image_content = create_overlay_image(image_content, merged_blocks, translations)
        overlay_image_base64 = base64.b64encode(overlay_image_content).decode('utf-8')

        return jsonify({
            'success': True,
            'original_texts': original_texts,
            'translations': translations,
            'overlay_image': overlay_image_base64,
            'text_regions_count': len(merged_blocks)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# In your app.py file, add this new route

@app.route('/translate-text-batch', methods=['POST'])
def translate_text_batch():
    """Translates a batch of text strings."""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'No text array provided'}), 400

        texts = data.get('texts')
        target_lang = data.get('target_lang', 'en')
        source_lang = data.get('source_lang', 'kn')

        if not isinstance(texts, list):
            return jsonify({'error': '"texts" must be an array'}), 400

        # Filter out empty or whitespace-only strings to avoid unnecessary API calls
        non_empty_texts = [text for text in texts if text and not text.isspace()]
        if not non_empty_texts:
            return jsonify({'success': True, 'translations': []})

        # Use the official client to translate the batch of texts
        # Note: The v2 client translates one by one, but batching is possible with v3
        translations = [translate_text(text, target_lang, source_lang) for text in texts]

        return jsonify({
            'success': True,
            'translations': translations
        })

    except Exception as e:
        # Log the full error for debugging on the server
        print(f"Error in /translate-text-batch: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'comic-translator'})


if __name__ == '__main__':
    # Make sure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
    # e.g., export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
    app.run(debug=True, host='0.0.0.0', port=5000)
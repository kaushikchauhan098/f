from flask import Flask, render_template, request
import requests
from PIL import Image
from io import BytesIO
from deepface import DeepFace
import base64
import os

app = Flask(__name__)

# Function to convert image from URL to JPG and return it as a BytesIO object
def get_image_from_url(url):
    try:
        if url.startswith('data:'):
            # Extract base64 encoded image data
            header, encoded = url.split(',', 1)
            data = base64.b64decode(encoded)

            # Open image from binary data
            img = Image.open(BytesIO(data))
        else:
            response = requests.get(url)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save the image to a BytesIO object
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
        return None

    except IOError as e:
        print(f"Error opening/saving image: {e}")
        return None

# Function to perform DeepFace matching, ensuring single face detection
def deepface_match(img1_bytes, img2_bytes):
    try:
        # Save BytesIO images to temporary paths for DeepFace
        img1_temp_path = 'temp_img1.jpg'
        img2_temp_path = 'temp_img2.jpg'

        with open(img1_temp_path, 'wb') as f1:
            f1.write(img1_bytes.getbuffer())
        
        with open(img2_temp_path, 'wb') as f2:
            f2.write(img2_bytes.getbuffer())

        # Perform DeepFace verification
        result = DeepFace.verify(img1_temp_path, img2_temp_path, detector_backend='mtcnn', enforce_detection=True)
        
        # Clean up temporary files
        os.remove(img1_temp_path)
        os.remove(img2_temp_path)
        
        return result['verified']

    except ValueError as e:
        print(f"DeepFace ValueError: {e}")
        return False

    except Exception as e:
        print(f"DeepFace Error: {e}")
        return False

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image1_url = None
    image2_url = None
    if request.method == 'POST':
        url1 = request.form['url1']
        url2 = request.form['url2']

        # Get images as BytesIO objects
        img1_bytes = get_image_from_url(url1)
        img2_bytes = get_image_from_url(url2)

        if img1_bytes and img2_bytes:
            # Perform DeepFace matching
            match_result = deepface_match(img1_bytes, img2_bytes)
            if match_result:
                result = "Faces Match"
            else:
                result = "Faces Do Not Match"

            image1_url = url1
            image2_url = url2
        else:
            result = "Failed to fetch or process one or both images. Please check the URLs and try again."

    return render_template('index.html', result=result, image1_url=image1_url, image2_url=image2_url)

if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

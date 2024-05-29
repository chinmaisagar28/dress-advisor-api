from flask import Flask, request, jsonify
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import base64
from flask_cors import CORS


# app = Flask(__name__)
app = Flask(__name__, static_folder='static')
CORS(app)

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

def removeBlack(estimator_labels, estimator_cluster):
    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)

def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):
    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=2, hasThresholding=False):
    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0, n_init=10)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    
    # Find the highest RGB values
    highestRed = max(color[0] for color in estimator.cluster_centers_)
    highestGreen = max(color[1] for color in estimator.cluster_centers_)
    highestBlue = max(color[2] for color in estimator.cluster_centers_)
    
    # Add highest RGB values to the dominantColors list
    dominantColors = [{'cluster_index': -1, 'color': [highestRed, highestGreen, highestBlue], 'color_percentage': 0}] + colorInformation
    
    return dominantColors

# Initialize arrays to store data and errors
colorsInfoArray = []  # Array to store color information
errors = []  # Array to store errors

def store_data(colors, image_base64, skin_base64, gender, occasion):
    # Create a dictionary to store the data
    data = {
        "image": image_base64,
        "skin": skin_base64,
        "dominant_colors" : colors,
        "gender" : gender,
        "occasion" : occasion,        
    }
    # Append the data to the array
    colorsInfoArray.append(data)

def store_error(err):
    # Create a dictionary to store the data
    err = {
        "error": {
            "message" : err
        }        
    }
    # Append the error to the array
    errors.append(err)

@app.route('/api/analyze', methods=['POST', 'GET'])
def analyze():
    if request.method == 'POST':
        try:
            # Clear existing data arrays
            colorsInfoArray.clear()
            errors.clear()
            
            # Get JSON data from the request
            data = request.json

            # Extract file data, gender, and occasion from JSON
            file_data = data['file']
            gender = data['gender']
            occasion = data['occasion']

            # Decode image data from Base64
            image_data = base64.b64decode(file_data.split(',')[1])

            # Decode image data into an OpenCV image
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                a = store_error('Unable to decode the image file. Please try with different File format!')
                return jsonify({'error': 'Unable to decode the image file. Please try with different File format!'}), 400

            # Resize image to a width of 550
            image = imutils.resize(image, width=550)

            # Detect faces in the uploaded image
            faces = detect_faces(image)

            # If no faces detected or multiple faces detected, render error message
            if len(faces) == 0:
                store_error('No faces detected in the uploaded image.')
                return jsonify({'error': 'No faces detected in the uploaded image.'}), 400
            elif len(faces) >= 2:
                store_error('Multiple Faces Detected')
                return jsonify({'error': 'Multiple Faces Detected'}), 400

            # Extract skin from the detected face region
            for (x, y, w, h) in faces:
                face_region = image[y:y+h, x:x+w]
                skin = extractSkin(face_region)

                # Find the dominant color in the skin region
                dominantColors = extractDominantColor(skin, hasThresholding=True)
                for color in dominantColors:
                    color['color'] = [int(val) for val in color['color']]
                print(dominantColors)

                # Encode images to Base64
                _, skin_encoded = cv2.imencode('.jpg', skin)
                skin_base64 = base64.b64encode(skin_encoded).decode('utf-8')

                _, image_encoded = cv2.imencode('.jpg', image)
                image_base64 = base64.b64encode(image_encoded).decode('utf-8')

                # Store image and other data if successful
                store_data(dominantColors, image_base64, skin_base64, gender, occasion)

                # Construct response data
                response_data = {
                    "image": image_base64,
                    "skin": skin_base64,
                    "dominant_colors": dominantColors,
                    "gender": gender,
                    "occasion": occasion
                }

                # Return the data as JSON response
                return jsonify(response_data), 200

        except Exception as e:
            # Handle exceptions
            store_error(f"An error occurred: {str(e)}. Please try again with a different image file.")
            return jsonify({"error": f"An error occurred: {str(e)}. Please try again with a different image file."}), 500
    if request.method == 'GET':
        return jsonify({"error":"you can't get that result directly"}), 400


@app.route('/api/result', methods=['GET'])
def result_api():
    if request.method == 'GET':
        # If request is GET, return the stored data or errors
        if errors:
            return jsonify(errors), 400
        else:
            return jsonify(colorsInfoArray), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({"colorsInfoArray":{"message":"This is index page"}}), 200



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)

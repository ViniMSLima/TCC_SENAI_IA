from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import sys

# Load the pre-trained model
model = tf.keras.models.load_model("flaskr/model3.keras")

bp = Blueprint('json', __name__, url_prefix='/json')

maybeResults = ["bad", "good"]


@bp.route('/', methods=['POST'])
def process_images():
    data = request.get_json()
    nImages = len(data['images'])
    wordResults = []

    for img_path in data['images']:
        img = img_path
        print(img)
        img_data = np.array([tf.keras.utils.load_img(img)])
        # Perform prediction using the loaded model
        wordResults.append(model.predict(img_data))

    results = ""
    for i in range(nImages):
        results += maybeResults[np.argmax(wordResults[i][0])]

    print(results)
    return jsonify(results)

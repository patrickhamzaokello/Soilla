from flask import Flask, render_template, url_for, redirect, request, flash, send_from_directory
from werkzeug.utils import secure_filename
import flask
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import io
import string
import time
from PIL import Image
from flask import Flask, jsonify, request

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

app.config['SECRET_KEY'] = '2008de4bbf105d61f26a763f8'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# image preprocessing and prediction.
def predict(imagepath, imagefilename):
    # some configvalues
    # set the path to the serialized model after training
    MODEL_PATH = os.path.sep.join(["model", "soil.model"])

    # initialize the list of class label names
    CLASSES = ["LoamSoil", "SandSoil", "ClaySoil"]

    # load the input image and then clone it so we can draw on it later
    image = cv2.imread(imagepath)
    originalimage = cv2.imread(imagepath)
    output = image.copy()
    output = imutils.resize(output, width=400)

    # our model was trained on RGB ordered images but OpenCV represents
    # images in BGR order, so swap the channels, and then resize to
    # 224x224 (the input dimensions for VGG16)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # convert the image to a floating point data type and perform mean
    # subtraction
    image = image.astype("float32")
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    image -= mean

    # load the trained model from disk
    print("[INFO] loading model...")
    model = load_model(MODEL_PATH)

    # pass the image through the network to obtain our predictions
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    i = np.argmax(preds)
    label = CLASSES[i]

    # results the prediction on the output image
    resultslabel = "{}".format(label)
    resultsaccuraccy = "{:.2f}%".format(preds[i] * 100)

    # load recommended plants
    if (label == "LoamSoil"):
        suggestion = "Loam is considered ideal for gardening and agricultural uses because it retains nutrients well and retains water while still allowing excess water to drain away.[4] A soil dominated by one or two of the three particle size groups can behave like loam if it has a strong granular structure, promoted by a high content of organic matter. However, a soil that meets the textural (geological) definition of loam can lose its characteristic desirable qualities when it is compacted, depleted of organic matter, or has clay dispersed throughout its fine-earth fraction."
    elif (label == "SandSoil"):
        suggestion = "grow some fucking Sand soil crops"
    elif (label == "ClaySoil"):
        suggestion = "grow some fucking Clay soil crops"
    else:
        suggestion = "Perform Test Again"

    # save the output image
    filename = imagefilename
    imageplacedpath = './static/uploads/' + filename

    cv2.imwrite(imageplacedpath, originalimage)
    print("written successfully")

    results = {
        "filename": filename,
        "resultslabel": resultslabel,
        "resultsaccuraccy": resultsaccuraccy,
        "country": "ug",
        "pointforhelp": {
            "contact": "0787250196",
            "location": "lira",
            "email": "pkasemer@gmail.com"
        },
        "suggested crops": {
            "intro": "suggested crops",
            "body": suggestion,
            "cropsheading": "suggested crops",
            "croplist": [
                "cropone {}".format(resultslabel),
                "croptwo {}".format(resultslabel),
                "cropthree {}".format(resultslabel),
                "cropfour {}".format(resultslabel)]
        }
    }

    return results


@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return jsonify("Please try again. The Image doesn't exist")

    file = request.files.get('file')

    if not file:
        return jsonify('No File Specified')

    if file.filename == '':
        return jsonify('No Image selected.', 'danger')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        prediction = predict(image, filename)
        return jsonify(prediction)
    else:
        return jsonify('Wrong file type', 'Allowed image types are: png, jpg, jpeg')


@app.route('/')
def upload_form():
    return render_template('main.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

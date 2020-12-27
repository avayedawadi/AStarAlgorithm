from flask import Flask, render_template, url_for, flash, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from project import main
import os
import cv2
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = ('jpg','jpeg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 


@app.route('/', methods = ['GET','POST'])
@app.route('/index', methods=['GET','POST'])
def show_index():
    return render_template("index.html")

@app.route('/success', methods = ['POST']) 
def success():  
    if request.method == 'POST':
        filelist = [f for f in os.listdir("static")]
        for f in filelist:
            os.remove(os.path.join("static", f))  
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
        images = [f for f in os.listdir("static")]
        print(images)
        image = cv2.imread("static/" + str(images[0]))
        img = keras.preprocessing.image.load_img(
            "static/" + str(images[0]), target_size=(180, 180)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        
        my_model = tf.keras.models.load_model('my_model/content/my_model')
        predictions = my_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        class_names = ['floorplan', 'random image', 'maze']
        model="The machine learning model predicts that this image is a {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
        print(
            "The machine learning model predicts that this image is a {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
            
        )
        morphSuggestion = ""
        if np.argmax(score) == 0:
            morphSuggestion = "Based on the fact that we predict this is a floorplan, we recommend enabling the morphological transformation below."
        elif np.argmax(score) == 1:
            morphSuggestion = "This is does not appear to be a floorplan or any discernible image. Therefore, please be prepared for unexpected results."
        elif np.argmax(score) == 2:
            moprhSuggestion = "This appears to be a maze. Therefore, we suggest you do not enable morphological transformations as mazes are not what our program is enabled to easily operate on."
        

        
        images = [f for f in os.listdir("static")]
        image = cv2.imread("static/" + str(images[0]))
        dim = None
        filename = str(time.time()) + str(images[0])
        if(image.shape[0] > 700):
            r = 700 / float(image.shape[0])
            dim = (int(image.shape[1] * r), 700)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            filelist = [f for f in os.listdir("static")]
            for f in filelist:
                os.remove(os.path.join("static", f))  
        cv2.imwrite("static/" + filename,image)
        return render_template("success.html", name = filename,user_image=filename,model=model,morphSuggest=morphSuggestion)

@app.route('/algorithm',methods=['POST'])
def algorithm():
    if request.method == 'POST':
        print(request.form['startingX'])
        print(request.form['startingY'])
        images = [f for f in os.listdir("static")]
        image = cv2.imread("static/" + str(images[0]))
        if(request.form['morph'] == "true"):
            morphBool = True
        else:
            morphBool = False
        path = main("static/" + str(images[0]), int(request.form['startingX']), int(request.form['startingY']), 
        int(request.form['goalX']), int(request.form['goalY']), morphBool)
        dim = None
        if(image.shape[0] > 700):
            r = 700 / float(image.shape[0])
            dim = (int(image.shape[1] * r), 700)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        for i in path:
            image[i[0],i[1]] = [0,0,255]
        filename = str(time.time()) + "image.jpg"
        filelist = [f for f in os.listdir("static")]
        for f in filelist:
            os.remove(os.path.join("static", f))
        cv2.imwrite("static/" + filename,image)
        #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'floorplan2.jpg')
        return render_template("algorithm.html",user_image=filename) 



if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, url_for, flash, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from project import main
import os
import cv2
import time

UPLOAD_FOLDER = "AStarProject/static"
ALLOWED_EXTENSIONS = ('jpg','jpeg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 


@app.route('/', methods = ['POST'])
@app.route('/index')
def show_index():
    return render_template("index.html")

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':
        filelist = [f for f in os.listdir("AStarProject/static")]
        for f in filelist:
            os.remove(os.path.join("AStarProject/static", f))  
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
        images = [f for f in os.listdir("AStarProject/static")]
        image = cv2.imread("AStarProject/static/" + str(images[0]))
        dim = None
        filename = str(time.time()) + str(images[0])
        if(image.shape[0] > 700):
            r = 700 / float(image.shape[0])
            dim = (int(image.shape[1] * r), 700)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            filelist = [f for f in os.listdir("AStarProject/static")]
            for f in filelist:
                os.remove(os.path.join("AStarProject/static", f))  
        cv2.imwrite("AStarProject/static/" + filename,image)
        return render_template("success.html", name = filename,user_image=filename)

@app.route('/algorithm',methods=['POST'])
def algorithm():
    if request.method == 'POST':
        print(request.form['startingX'])
        print(request.form['startingY'])
        images = [f for f in os.listdir("AStarProject/static")]
        path = main("AStarProject/static/" + str(images[0]), int(request.form['startingX']), int(request.form['startingY']), 
        int(request.form['goalX']), int(request.form['goalY']))
        image = cv2.imread("AStarProject/static/" + str(images[0]))
        dim = None
        if(image.shape[0] > 700):
            r = 700 / float(image.shape[0])
            dim = (int(image.shape[1] * r), 700)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        for i in path:
            image[i[0],i[1]] = [0,0,255]
        filename = str(time.time()) + "image.jpg"
        filelist = [f for f in os.listdir("AStarProject/static")]
        for f in filelist:
            os.remove(os.path.join("AStarProject/static", f))
        cv2.imwrite("AStarProject/static/" + filename,image)
        #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'floorplan2.jpg')
        return render_template("algorithm.html",user_image=filename) 



if __name__ == "__main__":
    app.run(debug=True)

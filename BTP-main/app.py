import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from matplotlib import image
from tensorflow import keras
from keras.models import model_from_json
from flask import Flask, render_template, request
import base64
import io
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = keras.models.load_model('crack_model.h5')
print(model.summary())
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#this part taken from documentation (line 24-33)
model.save_weights("model.h5")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("***")

@app.route("/",methods=['GET'])
def upload_file():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imgFile = request.files['imagefile']
    #we are saving the uploaded images in the static folder
    image_path = './static/'+imgFile.filename
    #now we are saving the image in the path specified above ...i.e static folder
    imgFile.save(image_path)
    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image = preprocess_input(image)
    yhat = loaded_model.predict(image)
    #label = decode_predictions(yhat)
    
    yhat[0][0]=round(yhat[0][0],2)
    yhat[0][1]=round(yhat[0][1],2)
    print(yhat[0][0])
    print(yhat[0][1])
    pred = ""
    if yhat[0][0]>=yhat[0][1]:
        pred = "This image is cracked"#str(yhat[0][0])
    else:
        pred = "This image is non cracked"#+str(yhat[0][1])
    return render_template('index.html',prediction=pred )

@app.route('/display',methods=['POST'])
def display():
    imgFile = request.files['imagefile']
    image_path = './static/'+imgFile.filename.split('.')[0]+'_disp.jpeg'
    imgFile.save(image_path)
    im = Image.open(image_path)
    data = io.BytesIO()
    im.save(data,"JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('index.html',img_data=encoded_img_data.decode('utf-8'))

@app.route('/process',methods=['POST'])
def process():
    #We are getting the uploaded image from the website
    imgFile = request.files['image']
    #we are saving the uploaded images in the static folder
    image_path = './static/'+imgFile.filename.split('.')[0]+'_process.jpeg'
    #now we are saving the image in the path specified above ...i.e static folder
    imgFile.save(image_path)
    img = cv2.imread(image_path)
    cv2.imshow("image uploaded",img)
    #applying the canny's algorithm and detecting the edges and defects in the uploaded image
    imgCanny = cv2.Canny(img,125,175)
    cv2.imshow("cracks",imgCanny)
    #now saving the processed image in the static folder with filename_processed.jpeg
    #the processed path is specified below
    processed_path = './static/'+imgFile.filename.split('.')[0]+'_processed.jpeg'
    #now for comparision between two images, we concatenate the original image and the processed image
    status = cv2.imwrite(processed_path,imgCanny)
    #the status variable above is a boolean variable
    #it tells us whether the image is saved properly or not
    imForConcat = cv2.imread(processed_path)
    im_h = cv2.hconcat([img,imForConcat])
    #we concatenated the two images horizontally for comparision
    #the concatenated image is updated in the processed path
    status = cv2.imwrite(processed_path,im_h)
    print("The status of writing the image is ",status)
    #now we are sending the concatenated image into the frontend for display
    im = Image.open(processed_path)
    data = io.BytesIO()
    im.save(data,"JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('index.html',img_data_process=encoded_img_data.decode('utf-8'))

@app.route('/count',methods=['POST'])
def count():
    imgFile = request.files['image']
    image_path = './static/'+imgFile.filename.split('.')[0]+'_crack.jpeg'
    imgFile.save(image_path)
    img = cv2.imread(image_path)
    #now convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #now we have to blur the images to reduce the noise
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    #now detecting the edges using the canny's algorithm
    canny = cv2.Canny(blur, 30, 150, 3)
    #now to thick the edges 
    dilated = cv2.dilate(canny, (1, 1), iterations=0)
    #now we have to count the contours
    (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    processed_path = './static/'+imgFile.filename.split('.')[0]+'_cprocessed.jpeg'
    status = cv2.imwrite(processed_path,rgb)
    print("The status of writing the image is ",status)
    im = Image.open(processed_path)
    data = io.BytesIO()
    im.save(data,"JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('index.html',img_data_processed=encoded_img_data.decode('utf-8'))

@app.route('/count_cracks',methods=['POST'])
def count_cracks():
    imgFile = request.files['image']
    image_path = './static/'+imgFile.filename.split('.')[0]+'_crack1.jpeg'
    imgFile.save(image_path)
    img = cv2.imread(image_path)
    #now convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #now we have to blur the images to reduce the noise
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    #now detecting the edges using the canny's algorithm
    canny = cv2.Canny(blur, 30, 150, 3)
    #now to thick the edges 
    dilated = cv2.dilate(canny, (1, 1), iterations=0)
    #now we have to count the contours
    (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    text="The approximate number of cracks in the image are "+str(len(cnt))
    return render_template('index.html',img_data_processed1=text)

if __name__=="__main__":
    app.run(debug=True)
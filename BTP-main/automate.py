import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from matplotlib import image
from tensorflow import keras
from keras.models import model_from_json
path = './test'
lst = os.listdir(path)

if os.path.exists('result.txt')==False:
    file1 = open("result.txt",'a')
    file1.write("The labels of the images that are cracked are: \n")
file1 = open("result.txt",'a')

def predict(path_of_image,loaded_model,index):
    image = load_img(path_of_image, target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image = preprocess_input(image)
    yhat = loaded_model.predict(image)
    yhat[0][0]=round(yhat[0][0],2)
    yhat[0][1]=round(yhat[0][1],2)
    if yhat[0][0]>=yhat[0][1]:
        st = "image "+str(index)+'\n'
        file1.write(st)


def model_init():
    model = keras.models.load_model('crack_model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
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
    return loaded_model

loaded_model = model_init()

for index, image in enumerate(lst):
    path_of_image = './test/'+image
    predict(path_of_image,loaded_model,index+1)



from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.views import View
import tensorflow as ts
import numpy as np
from io import BytesIO
from PIL import Image
import re
import base64
from django.shortcuts import render, redirect
from django.urls import reverse_lazy


# predicting number from canvas image

class num_rec(View):

    def get(self, request):
        response = TemplateResponse(request, 'main_page.html')
        return response

    def post(self, request):

        # get image
        img_size = 28, 28
        image_data = request.POST['image_data']
        image_width = int(request.POST['width'])
        image_height = int(request.POST['height'])

        # reshape image to base64
        image_data = re.sub("^data:image/png;base64,", "", image_data)
        image_data = base64.b64decode(image_data)
        image_data = BytesIO(image_data)
        im = Image.open(image_data)

        # rezise image from original (280x280) to modelled size (28x28)
        image = im.resize(img_size, Image.LANCZOS)
        # image = im.resize(img_size, Image.ANTIALIAS)

        # convert to numpy array
        image_array = np.asarray(image)

        image3 = []
        image2 = []

        # array dimension reduction
        for i in range(0, len(image_array)):
            tb = []

            for j in range(0, len(image_array[i])):
                # a=0
                # for k in range(0, len(image_array[i][j])-1):
                #     a=a+image_array[i][j][k]
                # a=a/4
                a = (image_array[i][j][3] * 2 + 0 * image_array[i][j][2]) / 2
                tb.append(a)
            image2.append(tb)

        # ṃultiply observations
        for _ in range(30):
            image3.append(image2)

        # calling ML model and predicting number from image
        new_model = ts.keras.models.load_model('number_recognition.model')
        predict = new_model.predict(image3)
        response1=(np.argmax(predict[0]))

        ctx = {
            "number": response1
        }
        return render(request, "result.html", ctx)

        # return HttpResponse(predict[0])


# model creation
def create_model():
    mnist = ts.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = ts.keras.utils.normalize(x_train, axis=1)
    x_test = ts.keras.utils.normalize(x_test, axis=1)
    model = ts.keras.models.Sequential()
    model.add(ts.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(ts.keras.layers.Dense(128, activation=ts.nn.relu))
    model.add(ts.keras.layers.Dense(128, activation=ts.nn.relu))
    model.add(ts.keras.layers.Dense(10, activation=ts.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    model.save('number_recognition.model')

# testing in python console
# new_model = ts.keras.models.load_model('number_recognition.model')
# predict = new_model.predict([x_test])
# import numpy as np
# print(np.argmax(predict[5]))

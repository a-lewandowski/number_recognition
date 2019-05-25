import jsonify as jsonify
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.views import View
import tensorflow as ts
import numpy as np
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django import forms
from io import BytesIO
from PIL import Image
import re
import base64
#
#
# class ImageUploadForm(forms.Form):
#     """Image upload form."""
#     image = forms.ImageField()
#
#
# class UploadFileForm(forms.Form):
#     title = forms.CharField(max_length=50)
#     file = forms.FileField()


# Create your views here.

class num_rec(View):

    def get(self, request):
        response = TemplateResponse(request, 'main_page.html')
        return response

    def post(self, request):
        img_size = 28, 28
        # im_h=28
        # print(request.POST.keys())
        # import pprint
        # pprint.pprint(request.POST  )
        image_data = request.POST['image_data']
        image_width = int(request.POST['width'])
        image_height = int(request.POST['height'])
        image_data = re.sub("^data:image/png;base64,", "", image_data)
        image_data = base64.b64decode(image_data)
        image_data = BytesIO(image_data)
        im = Image.open(image_data)

        image = im.resize(img_size, Image.LANCZOS)
        # image = im.resize(img_size, Image.ANTIALIAS)
        image_array = np.asarray(image)

        image3 = []
        image2 = []

        for i in range(0, len(image_array)):
            tb = []

            for j in range(0, len(image_array[i])):
                # a=0
                # for k in range(0, len(image_array[i][j])-1):
                #     a=a+image_array[i][j][k]
                # a=a/4
                a = image_array[i][j][3]
                tb.append(a)
            image2.append(tb)
        #
        # # image_array = image_array.flatten()
        # # image_array = image_array.reshape(1, 28, 28)
        image3.append(image2)
        image3.append(image2)

        # # image3 = np.asarray(image3)
        # image3 = ts.keras.utils.normalize(image3, axis=1)
        #
        # # with ts.Session() as sess:
        #     saver = ts.train.import_meta_graph('tmp/tensor_model.meta')
        #     saver.restore(sess, ts.train.latest_checkpoint('tmp/'))
        #
        #     predict_number = ts.argmax(ten.y, 1)
        #     predicted_number = ten.sess.run([predict_number], feed_dict={ten.x: [image_array]})
        #     guess = predicted_number[0][0]
        #     guess = int(guess)
        #     print(guess)
        #
        # return jsonify(guess=guess)  # returns as jason format

        # img = ts.decode_base64(image_data)

        from keras.preprocessing.image import img_to_array, array_to_img
        from keras.applications.vgg16 import preprocess_input, VGG16
        from keras.models import Model
        # im = im.resize((im_h, im_h), Image.ANTIALIAS)
        # img = np.invert(im.convert('L')).ravel()
        # image = Image.open(BytesIO(base64.b64decode(base64_image_string)))
        # img = np.array(im)

        # x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h, im_h))) for im in image_data])
        # img = preprocess_input(img)
        # img = image_data.img_to_array(img)
        new_model = ts.keras.models.load_model('number_recognition.model')
        # imag = ts.keras.img_to_array(image_data)
        # test_img = imag.reshape((1,784))
        predict = new_model.predict(image3)
        # predict = image2
        # predict = image_array
        # import numpy as np
        response1=(np.argmax(predict[0]))

        # response1 = guess
        response = HttpResponse(response1)

        # return response
        # else:
        # response = HttpResponse('error')

        return response

        # color = 'red'
        # color = request.POST('color')
        # if form.is_valid():
        #     # color = form.cleaned_data.get('color')
        #     ctx = {'form': form, 'color': color}
        #     return render(request, "set_colors.html", ctx)

        # response = HttpResponse('success')
        # return response


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

# new_model = ts.keras.models.load_model('number_recognition.model')
# predict = new_model.predict([x_test])
# import numpy as np
# print(np.argmax(predict[5]))

# img = image.load_img(path="testimage.png",grayscale=True,target_size=(28,28,1))
# img = image.img_to_array(img)
# test_img = img.reshape((1,784))
#
# img_class = model.predict_classes(test_img)
# prediction = img_class[0]
#
# classname = img_class[0]
#
# print("Class: ",classname)
# img = img.reshape((28,28))
# plt.imshow(img)
# plt.title(classname)
# plt.show()

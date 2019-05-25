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

class ImageUploadForm(forms.Form):
    """Image upload form."""
    image = forms.ImageField()


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()

# Create your views here.

class num_rec(View):

    def get(self, request):
        response = TemplateResponse(request, 'main_page.html')
        return response

    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)
        import pprint
        pprint.pprint(form)
        if form.is_valid():
            # handle_uploaded_file(request.FILES['file'])
            # return HttpResponseRedirect('/success/url/')
            response = HttpResponse('success')
            # return response
        else:
            response = HttpResponse('error')

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
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
import pickle
import cv2
import time


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

filename = 'model.pkl'
model = pickle.load(open(filename,'rb'))
print('model loaded')

def predict(img):
    img=cv2.imread(img)
    img_resize=cv2.resize(img,(150,150))
    l=[img_resize.flatten()]
    pre=model.predict(l)
    return pre[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        try:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                uploaded_file.save('static/input.jpg')
                pre=predict('static/input.jpg')
                if pre==1:
                    res='You have Lung Cancer'
                else:
                    res='Your Lung is Healthy'
                time.sleep(2)
                return render_template('predict.html',pre=res)
        except:
            pass
        else:
            return 'some error'


if __name__ == '__main__':
    app.run(debug=True)
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
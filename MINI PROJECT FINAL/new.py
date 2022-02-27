import pickle
import cv2

filename = 'model.pkl'
model = pickle.load(open(filename,'rb'))
print('model loaded')

def predict(img):
    img=cv2.imread(img)
    img_resize=cv2.resize(img,(150,150))
    l=[img_resize.flatten()]
    pre=model.predict(l)
    print("The predicted image is : ",pre[0])
predict('img.jpg')
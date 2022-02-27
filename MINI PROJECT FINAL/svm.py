import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
Categories=['no','yes']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir='C:/Users/Dell/OneDrive/Desktop/liver_data 1/train/'

for i in Categories:
    
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data


from sklearn import svm
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

#model=SVC(kernel="linear")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV 


import pickle

filename = 'C:/Users/Dell/OneDrive/Desktop/liver_data 1/model.pkl'
pickle.dump(model, open(filename, 'wb'))


loaded_model = pickle.load(open('C:/Users/Dell/OneDrive/Desktop/liver_data 1/model.pkl', 'rb'))

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
acc_score=accuracy_score(y_pred,y_test)
print(confusion_matrix(y_test, y_pred))

import cv2
img=cv2.imread('C:/Users/Dell/OneDrive/Desktop/liver_data 1/test/no/1.jpg')
plt.imshow(img)
plt.show()
img_resize=cv2.resize(img,(150,150))
l=[img_resize.flatten()]
probability=loaded_model.predict(l)
print("The predicted image is : "+Categories[model.predict(l)[0]])
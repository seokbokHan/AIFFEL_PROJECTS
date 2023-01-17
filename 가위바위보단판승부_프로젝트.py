# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:09:37 2023

@author: sbhan
"""

from PIL import Image
import os, glob
import matplotlib.pyplot as plt
import numpy as np
#%%
#%matplotlib inline
image_dir_path = 'D:\\AIFFEL\\rock_scissor_paper\\paper\\'
image_pil = Image.open(image_dir_path + 'papers1.jpg')
image = np.array(image_pil)

plt.imshow(image)
plt.show()
#%%
image.shape
#%%
def resize_images(img_path):
    images=glob.glob(img_path + "\\*.jpg")
    
    print(len(image), " images to be resized")
    
    # 파일마다 모두 28x28 사이즈로 바꾸어 저장.
    target_size=(28, 28)
    for img in images:
        old_img = Image.open(img)
        new_img = old_img.resize(target_size, Image.ANTIALIAS)
        new_img.save(img, "JPEG")
        
    print(len(images), " images resized.")
#%%
#가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어 들여서
image_dir_path = "D:\\AIFFEL\\rock_scissor_paper\\scissor"
resize_images(image_dir_path)

print("가위 이미지 resize 완료!")
#%%
#바위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어 들여서
image_dir_path = "D:\\AIFFEL\\rock_scissor_paper\\rock"
resize_images(image_dir_path)

print("바위 이미지 resize 완료!")
#%%
image_dir_path = "D:\\AIFFEL\\rock_scissor_paper\\paper"
resize_images(image_dir_path)

print("보 이미지 resize 완료!")
#%%
def load_data(img_path, number_of_data=3180):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size, color = 28, 3

    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'\\scissor\\*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'\\rock\\*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'\\paper\\*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
    return imgs, labels

image_dir_path = "D:\\AIFFEL\\rock_scissor_paper\\" #os.getenv("HOME") + "/aiffel/rock_scissor_paper/data/"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
        
#%%
plt.imshow(x_train[0])
print('라벨: ', y_train[0])
#%%
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
#%%
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')    
    ])

model.summary()
#%%
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
#%%
model.fit(x_train, y_train, epochs=100)
#%%
def load_test(img_path, number_of_data=3180):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size, color = 28, 3

    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'\\test\\scissor\\*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'\\test\\rock\\*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'\\test\\paper\\*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
    return imgs, labels
        
#%%
#가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어 들여서
image_dir_path = "D:\\AIFFEL\\rock_scissor_paper\\test\\scissor"
resize_images(image_dir_path)
print("가위 이미지 resize 완료!")

#바위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어 들여서
image_dir_path = "D:\\AIFFEL\\rock_scissor_paper\\test\\rock"
resize_images(image_dir_path)
print("바위 이미지 resize 완료!")

#보 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어 들여서
image_dir_path = "D:\\AIFFEL\\rock_scissor_paper\\test\\paper"
resize_images(image_dir_path)
print("보 이미지 resize 완료!")        
#%%
image_dir_path = "D:\\AIFFEL\\rock_scissor_paper\\" #os.getenv("HOME") + "/aiffel/rock_scissor_paper/data/"
(x_test, y_test)=load_test(image_dir_path)
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화
print("x_test shape: {}".format(x_train.shape))
print("y_test shape: {}".format(y_train.shape))
#%%
model.evaluate(x_test_norm, y_test)

                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
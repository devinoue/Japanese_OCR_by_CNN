
# coding: utf-8

# In[ ]:


import numpy as np
import cv2,pickle
import sys
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,GlobalAveragePooling2D,Input
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('tf')

# データファイルと画像サイズの指定
data_file = "./nihon_go.pickle"
im_size=32
out_size=956 #出力層の要素数(インデックスではなく要素の個数)
im_color =3 # 画像の色空間/グレースケール
in_shape = (im_size, im_size, im_color)

n_categories=957

# カタカナ画像のデータセットを読み込む
data = pickle.load(open(data_file, "rb"))

# 画像データを0-1の範囲に直す
y=[]
x=[]

for d in data:
	(num, img) = d
	img = img.astype("float").reshape(im_size,im_size,im_color)/255
	img=img.transpose(2,0,1)
	y.append(keras.utils.np_utils.to_categorical(num,957))
	x.append(img)


x=np.array(x)
y=np.array(y)

# print(x.shape)
# sys.exit()


# 学習用とテスト用に分離
x_train,x_test,y_train,y_test = train_test_split(
	x,y,test_size=0.2,train_size=0.8
)

base_model = VGG16(include_top=False,weights='imagenet', input_shape=(in_shape))
x=base_model.output
x=GlobalAveragePooling2D()(x)
prediction = Dense(n_categories,activation='softmax')(x)

model = Model(input=base_model.input,outputs=prediction)

for layer in base_model.layers[:15]:
    layer.trainable=False


# モデルをコンパイルして学習を実行
model.compile(
	loss="categorical_crossentropy",
	optimizer=SGD(lr=0.0001,momentun=0.9),
	metrics=['accuracy']
)
print("compliled")
#学習して評価
hist=model.fit(
	x_train,y_train,
	batch_size=128,epochs=12,verbose=1,
	validation_data=(x_test,y_test)
)

# モデルの評価
score = model.evaluate(x_test,y_test,verbose=1)
print("正答率=",score[1],"loss=",score[0])

# 学習の様子をプロット
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy')
plt.legend(['train','test'],loc='upper left')
plt.show()

#ロスの推移をプロット
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train','test'],loc='upper left')
plt.show()



# coding: utf-8

# In[12]:


import numpy as np
import cv2,pickle
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt


# データファイルと画像サイズの指定
data_file = "./nihon_go.pickle"
im_size=25
out_size=956 #出力層の要素数(インデックスではなく要素の個数)
im_color =1 # 画像の色空間/グレースケール
in_shape = (im_size, im_size, im_color)

# カタカナ画像のデータセットを読み込む
data = pickle.load(open(data_file, "rb"))

# 画像データを0-1の範囲に直す
y=[]
x=[]

for d in data:
	(num, img) = d
	img = img.astype("float").reshape(
		im_size,im_size,im_color)/255

	y.append(keras.utils.np_utils.to_categorical(num,957))
	x.append(img)

x=np.array(x)
y=np.array(y)

# 学習用とテスト用に分離
x_train,x_test,y_train,y_test = train_test_split(
	x,y,test_size=0.2,train_size=0.8
)

# CNNモデル構築を定義
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=in_shape))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(957,activation="softmax"))


# モデルをコンパイルして学習を実行
model.compile(
	loss="categorical_crossentropy",
	optimizer=RMSprop(),
	metrics=['accuracy']
)
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


# In[15]:


import pickle
# ラベルと画像のデータを保存*4
pickle.dump("plt", open("save_plot.pickle","wb"))


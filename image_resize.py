
# coding: utf-8

# In[1]:


import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sys


# 保存先や画像サイズの指定
out_dir="./extract"#画像データのあるディレクトリ
im_size = 32 #画像サイズ
save_file = "nihon_go.pickle" #保存先


# 参考 : https://qiita.com/SKYS/items/cbde3775e2143cad7455
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


result=[]
j=0
img_dirs = glob.glob(out_dir + "/*")
for i,dir in enumerate(img_dirs):
    _,ocr_str = dir.split("\\")
    print(str(i)," : ",ocr_str)

    
    img_files = glob.glob(dir + "/*")
    for img_path in img_files:
        img = imread(img_path)
        if all(img[0][0] == None):
            continue
        img = cv2.resize(img,(im_size,im_size))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         j+=1
#         if j <= 10:
            
#             plt.subplot(2,5,j)
#             plt.imshow(img,cmap='gray')
#         else:
#             plt.show()
#             sys.exit()
        result.append([i,img])

# ラベルと画像のデータを保存*4
pickle.dump(result, open(save_file,"wb"))

print("ok")




# In[ ]:


_


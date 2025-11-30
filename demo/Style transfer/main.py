import torch.cuda
from PIL import Image
from preprocessing import train,save
import os
from glob import glob

if __name__ == '__main__':
    print("training...")
    os.makedirs("./Storage",exist_ok=True)
    configue={"lr":0.003,
              "epochs":1000,
              "device":"cuda" if torch.cuda.is_available()else "cpu",
              "steps":200, # 每steps就 将学习率变为当前的 0.8
              "image_shape":(512 , 768),
              "epoch_step":300}# 每 epoch_step 就保存一次图片
    content_image=Image.open(glob("./Image/content*")[0]) # 300x168
    style_img=Image.open(glob("./Image/style*")[0])  #259x194

    X,imgs=train(configue,content_image,style_img)
    if input("是否保存图像,1为保存，否则不保存")=="1":
        save(imgs,configue)
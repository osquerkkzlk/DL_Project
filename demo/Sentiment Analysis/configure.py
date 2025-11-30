import time

def configure():
    while True:
        x1=input("选择本次训练所用的模型   1:LSTM      0:TextCNN\n")
        x2=input("选择配置方式 :     1:is_show   0:is_not_show \n")
        x3=input("选择配置方式 :     1:is_train   0:is_not_train \n")
        x4=input("选择配置方式 :     1:is_predict   0:is_not_predict \n")
        if x1 in["1","0"] and x2 in["1","0"] and x3 in["1","0"] and x4 in["1","0"]:
            break
        else:
            print("----------------waiting----------------")
            time.sleep(3)
    return [int(x1),int(x2),int(x3),int(x4)]


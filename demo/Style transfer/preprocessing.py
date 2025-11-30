import torch
from torch import nn
import torchvision
import tqdm
import matplotlib.pyplot as plt


mean=torch.tensor([0.485,0.456,0.406])
std=torch.tensor([0.229,0.224,0.225])
loss=nn.MSELoss()

def preprocess(img,img_shape=(224,224)):
    transformers=torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_shape),
        torchvision.transforms.ToTensor(), # 转换通道、转换类型float32、/255归一化
        torchvision.transforms.Normalize(mean=mean,std=std)
    ])
    return transformers(img).unsqueeze(0)

def deprocess(img):
    # shape:(1,C,H,W)
    img=img[0].to(mean.device)
    img=torch.clamp(img.permute(1,2,0)*std+mean,0,1)
    return torchvision.transforms.ToPILImage()(img.permute(2,0,1))

def extract_features(net,X,layers):
    # layers=[style_layers,content_layers]
    # content_pred , style_pred
    style_pred,content_pred=[],[]
    max_value=max(max(layers[0]),max(layers[1]))
    for i in range(max_value+1):
        X=net[i](X)
        if i in layers[0]:
            style_pred.append(X)
        elif i in layers[1]:
            content_pred.append(X)
    return style_pred,content_pred


def get(content_image,style_img,device,image_shape=(224,224)):
    print("loading...")
    net = torchvision.models.vgg19(weights='IMAGENET1K_V1').features.to(device)
    print("VGG19'Architecture\n",net)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    style_layers,content_layers=[0,5,10,19,28],[25]
    layers=[style_layers,content_layers]

    print("transforming...")
    content_X=preprocess(content_image,image_shape).to(device)
    style_X=preprocess(style_img,image_shape).to(device)

    return content_X,style_X,layers,net

def content_loss(y_pred,y):
    # 我们不希望改变原始参数
    return loss(y_pred,y.detach())

def gram(X):
    X=X.reshape(X.shape[1],-1)
    return X@X.T / X.numel()

def style_loss(y_pred,gram_y):
    # 提前做了gram，初始值本来就是不变的，减小计算量
    return loss(gram(y_pred),gram_y.detach())

def tv_loss(y):
    # total variation denoising
    return 0.5*(torch.abs(y[:,:,1:,:]-y[:,:,:-1,:]).mean()+\
                torch.abs(y[:,:,:,1:]-y[:,:,:,:-1]).mean())

def criterion(X,contents_X,contents_pred,style_X,style_pred,\
              weight={"content":0.01,"style":1e5,"tv":10}):
    contents_l=sum([content_loss(y_pred,y)for y_pred,y in zip(contents_pred,contents_X)])
    style_l=sum([style_loss(y_pred,y)for y_pred,y in zip(style_pred,style_X)])
    tv_l=tv_loss(X)
    sum_l=weight["content"]*contents_l+weight["style"]*style_l+weight["tv"]*tv_l
    return contents_l,style_l,tv_l,sum_l

def init(X,lr,device):
    class Up_Image(nn.Module):
        def __init__(self,X,device,**kwargs):
            super().__init__(**kwargs)
            self.X=nn.Parameter(torch.randn(X.shape,device=device))
            self.X.data.copy_(X)
        def forward(self):
            return self.X
    img=Up_Image(X,device)
    return img(),torch.optim.AdamW(img.parameters(),lr=lr)

class Recorder:
    def __init__(self,num):
        self.metric=[[]for _ in range(num)]

    def __getitem__(self, item):
        return self.metric[item]
    def add(self,*args):
        if args:
            for i,arg in enumerate(args):
                self.metric[i].append(arg)

def train(configue,content_img,style_img):
    # 图像基本的预处理，转换成张量
    content_X, style_X, layers,net = get(content_img, style_img, configue["device"],configue["image_shape"])

    with torch.no_grad():
        #提取特征
        _,content_features=extract_features(net, content_X, layers)
        style_features,_=extract_features(net, style_X, layers)
        style_gram=[gram(x) for x in style_features]

    # 初始化图像
    X=content_X.clone().detach().requires_grad_(True)
    optim=torch.optim.AdamW([X], lr=configue["lr"])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=0.8,step_size=configue["steps"])
    pbar=tqdm.tqdm(total=configue["epochs"],desc="training...")

    loss_recorder=Recorder(4)
    img_recorder=Recorder(1)

    for epoch in range(configue["epochs"]):
        optim.zero_grad()
        style_pred,content_pred=extract_features(net,X,layers)
        contents_l, style_l, tv_l, sum_l=criterion(X,content_features,content_pred,style_gram,style_pred)
        sum_l.backward()
        optim.step()
        scheduler.step()
        if epoch % configue["epoch_step"]==0:
            img_recorder.add(X)
        pbar.update(1)
        pbar.set_description(f"<  loss  >{sum_l}")
        loss_recorder.add(contents_l.item(),style_l.item(),tv_l.item(),sum_l.item())

    dispaly_loss(loss_recorder[0],loss_recorder[1],loss_recorder[2],loss_recorder[3])
    return X,img_recorder[0]

def dispaly_loss(contents_l,style_l,tv_l,sum_l):
    def display_save(loss, style, label):
        plt.plot(range(1, 1 + len(loss)), loss, style, label=label)

    display_save(contents_l,"r","contents loss")
    display_save(style_l,"b","style_l loss")
    display_save(tv_l,"c","tv_l loss")
    display_save(sum_l,"m","sum_ loss")
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss value")
    plt.title("loss curve")
    plt.savefig("./Storage/loss curve.png")
    plt.show()

    print(f"content loss:{contents_l[-1]:-30}")
    print(f"style_l loss:{style_l[-1]:-30}:")
    print(f"tv_l loss:{tv_l[-1]:-30}")
    print(f"sum_l loss:{sum_l[-1]:-30}")


def save(imgs,configue):
    print("saving...")
    for i,img in enumerate(imgs):
        temp_img=deprocess(img)
        temp_img.save(f"./Storage/{(i+1)*configue['epoch_step']}_{configue['epochs']}.png")

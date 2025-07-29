import jittor as jt
import pandas as pd
import numpy as np
import Img_Dataset
from jittor.dataset import DataLoader
from jittor import nn
import matplotlib.pyplot as plt
import copy
import time
## dataloader
jt.flags.use_cuda=1
batch_size=100
train_set,val_set=Img_Dataset.build_dataset()

train_loader=DataLoader(dataset=train_set,batch_size=batch_size)
# for batch_id,(batch_img,target) in enumerate(train_loader):
#     print(batch_img.shape)
#     break
# for X,y in train_loader:
#     print(X.shape,y)
#     break
val_loader=DataLoader(dataset=val_set,batch_size=batch_size)

# conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
conv_arch = ((1, 64), (4, 128), (6, 256), (16, 512), (1, 512))
# conv_arch = ((1, 64), (2, 128), (4, 256), (14, 512), (1, 512))
ratio=4
small_conv_arch=[(pair[0],pair[1]//ratio) for pair in conv_arch]
## net

# def vgg_block(num_convs, in_channels, out_channels):
#     layers = []
#     for _ in range(num_convs):
#         layers.append(nn.Conv2d(in_channels, out_channels,
#                                 kernel_size=3, padding=1))
#         layers.append(nn.ReLU())
#         layers.append(nn.BatchNorm2d(out_channels))
#         in_channels = out_channels
#     layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
#     return nn.Sequential(*layers)


# def vgg(conv_arch):
#     conv_blks = []
#     in_channels = 3
#     for (num_convs, out_channels) in conv_arch:
#         conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
#         in_channels = out_channels

#     return nn.Sequential(
#         *conv_blks, nn.Flatten(),
#         nn.Linear(out_channels * 3 * 3, 100)# , nn.ReLU(), nn.Dropout(0.5),
#         # nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
#         # nn.Linear(4096, 100))
#     )

# net = vgg(small_conv_arch)

class RepVGG_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1,deploy=False):
        super().__init__()
        self.deploy=deploy
        self.rbr_reparam=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.rbr_identity=nn.BatchNorm2d(in_channels) if in_channels==out_channels and stride==1 else None
        self.rbr_dense=nn.Sequential()
        self.rbr_dense.add_module('conv',nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=stride,bias=False))
        self.rbr_dense.add_module('bn',nn.BatchNorm2d(out_channels))
        self.rbr_1x1=nn.Sequential()
        self.rbr_1x1.add_module('conv',nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=padding-kernel_size//2,stride=stride,bias=False))
        self.rbr_1x1.add_module('bn',nn.BatchNorm2d(out_channels))
    def execute(self,inputs):
        if self.deploy:
            return nn.relu(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out=0
        else:
            id_out=self.rbr_identity(inputs)
        return nn.relu(self.rbr_dense(inputs)+self.rbr_1x1(inputs)+id_out)
    def fuse_bn_Var(self,branch):
        pass
        if branch==None:
            return 0,0
        if isinstance(branch,nn.Sequential):
            if branch.conv.kernel_size==(1,1):
                kernel=nn.pad(branch.conv.weight,[1,1,1,1])
            else:
                kernel=branch.conv.weight
            running_mean=branch.bn.running_mean
            running_var=branch.bn.running_var
            gamma=branch.bn.weight
            beta=branch.bn.bias
            eps=branch.bn.eps
        elif isinstance(branch,nn.BatchNorm2d):
            pass
            kernel=jt.zeros((self.rbr_dense.conv.out_channels,self.rbr_dense.conv.in_channels,3,3))
            for i in range(kernel.shape[0]):
                kernel[i,i,1,1]=1
            running_mean=branch.running_mean
            running_var=branch.running_var
            gamma=branch.weight
            beta=branch.bias
            eps=branch.eps
        return kernel*(gamma/(jt.sqrt(running_var+eps))).reshape((-1,1,1,1)),beta-running_mean*gamma/(jt.sqrt(running_var+eps))
    def switch_to_deploy(self):
        pass
        weight_1x1,bias_1x1=self.fuse_bn_Var(self.rbr_1x1)
        weight_dense,bias_dense=self.fuse_bn_Var(self.rbr_dense)
        weight_id,bias_id=self.fuse_bn_Var(self.rbr_identity)
        self.rbr_reparam.weight.data=weight_1x1+weight_dense+weight_id
        self.rbr_reparam.bias.data=bias_1x1+bias_dense+bias_id
        # print(self.rbr_reparam(inputs)-self.rbr_1x1(inputs))
        # print(self.rbr_reparam.weight==self.rbr_dense.conv.weight)
        self.deploy=True
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self,'rbr_identity'):
            self.__delattr__('rbr_identity')
        

# blk=RepVGG_block(in_channels=5,out_channels=5)
# blk.eval()
# a=jt.randn((1,5,12,12))
# print(blk(a)-blk.switch(a))
# O=blk.switch(a)

class RepVGG(nn.Module):
    def __init__(self,conv_arch):
        super().__init__()
        self.conv_arch=conv_arch
        conv_blocks=[]
        in_channels=3
        for (num_convs,out_channels) in self.conv_arch:
            conv_blocks.append(self.make_stage(num_convs,in_channels,out_channels,stride=2))
            in_channels=out_channels
            self.out_channels=out_channels
        self.conv_blocks=nn.Sequential(*conv_blocks)
        self.pool_block=nn.AdaptiveAvgPool2d(output_size=1)
        self.dense_block=nn.Linear(self.out_channels,100)
        
    def make_stage(self,num_convs,in_channels,out_channels,stride):
        block=[]
        strides=[stride]+[1]*(num_convs-1)
        for each_stride in strides:
            block.append(RepVGG_block(in_channels,out_channels,stride=each_stride))
            in_channels=out_channels
        return nn.Sequential(*block)
    
    def execute(self,inputs):
        outputs=self.conv_blocks(inputs)
        outputs=self.pool_block(outputs)
        outputs=outputs.view(outputs.size(0),-1)
        outputs=self.dense_block(outputs)
        return outputs
    
    def switch_to_deploy(self):
        for stage in self.conv_blocks:
            for blk in stage:
                blk.switch_to_deploy()

net=RepVGG(conv_arch)

# print(net.modules)
## train

def accuracy(y_hat,y):
    y_hat,_=jt.argmax(y_hat,dim=1)
    cmp=(y_hat.data==y.data)
    return float(cmp.sum()/len(y.data))

def top5_accuracy(y_hat,y):
    y_hat=np.array(y_hat).argsort(axis=1)
    c=0
    for i in range(y_hat.shape[0]):
        if y[i] in y_hat[i][-5:]:
            c+=1
    return float(c/y_hat.shape[0])

class Earlystopper:
    def __init__(self,patience=5,min_delta=0.0):
        self.patience=patience
        self.min_delta=min_delta
        self.stop=False
        self.best_loss=float('inf')
        self.counter=0
        self.best_model_state=None
    def __call__(self,val_loss,model):
        if val_loss<self.best_loss-self.min_delta:
            self.best_loss=val_loss
            self.counter=0
            self.best_model_state=copy.deepcopy(model.state_dict())
        else:
            self.counter+=1
            if self.counter>self.patience:
                self.stop=True
                print(f'No improvement for {self.patience} epochs')
    def restore(self,model):
        model.load_state_dict(self.best_model_state)
        
loss=nn.CrossEntropyLoss()
trainer=jt.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0.05)
num_epoch=50
train_loss_total=[]
val_loss_total=[]
early_stopper=Earlystopper(patience=10,min_delta=1e-3)
origin_deploy_time_list=[]
for epoch in range(num_epoch):
    train_loss,val_loss=0,0
    train_acc,top1_val_acc,top5_val_acc=0,0,0
    net.train()
    for X,y in train_loader:
        trainer.zero_grad()
        l=loss(net(X),y)
        trainer.backward(l)
        trainer.step()
    with jt.no_grad():
        net.eval()
        for X,y in val_loader:
            top1_val_acc+=accuracy(net(X),y)
            top5_val_acc+=top5_accuracy(net(X),y)
            val_loss+=loss(net(X),y)
             
        for X,y in train_loader:
            train_acc+=accuracy(net(X),y)
            train_loss+=loss(net(X),y)
        
        top1_val_acc/=val_set.total_len/batch_size
        top5_val_acc/=val_set.total_len/batch_size
        train_acc/=train_set.total_len/batch_size
        
        val_loss/=val_set.total_len/batch_size
        train_loss/=train_set.total_len/batch_size
        
        train_loss_total.append(train_loss)
        val_loss_total.append(val_loss)
        
        print(f'epoch: {epoch+1}')
        print(f'train_loss: {train_loss}, val_loss: {val_loss}')
        print(f'train_acc: {train_acc}, top1_val_acc: {top1_val_acc}, top5_val_acc: {top5_val_acc}')

        early_stopper(val_loss,net)
        if early_stopper.stop:
            break

early_stopper.restore(net)
jt.save(net.state_dict(),'./train_weights.p')

net.eval()
top1_val_acc,top5_val_acc=0,0
start_time=time.time()
for X,y in val_loader:
    top1_val_acc+=accuracy(net(X),y)
    top5_val_acc+=top5_accuracy(net(X),y)
    val_loss+=loss(net(X),y)
end_time=time.time()
print(f'origin_deploy_time: {end_time-start_time}')

net.switch_to_deploy()
jt.save(net.state_dict(),'./deploy_weights.p')
top1_val_acc,top5_val_acc=0,0
start_time=time.time()
for X,y in val_loader:
    top1_val_acc+=accuracy(net(X),y)
    top5_val_acc+=top5_accuracy(net(X),y)
    val_loss+=loss(net(X),y)
end_time=time.time()
print(f'rep_deploy_time: {end_time-start_time}')
print(f'final_top1_acc: {top1_val_acc},final_top5_acc: {top5_val_acc}')


plt.figure(figsize=(10, 5))
plt.plot(train_loss_total,label='Train Loss')
plt.plot(val_loss_total,label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('VGG11_bn Loss Curve')
plt.title('RepVGG Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('RepVGG Loss Curve.png')
plt.show()

# print(train_set.class_to_idx)


# for X,y in train_loader:
#     print(X.shape,y)
#     print(net(X).shape)
#     break
# print(train_set.total_len,val_set.total_len)

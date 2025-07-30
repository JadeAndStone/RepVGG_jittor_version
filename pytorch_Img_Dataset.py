import torch
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import os
import shutil
import random

def build_dataset():
    data_dir='./images'
    new_data_dir='./images_processed'
    train_df_dir='./train.csv'
    val_df_dir='./val.csv'
    test_df_dir='./test.csv'
    train_df=pd.read_csv(train_df_dir)
    val_df=pd.read_csv(val_df_dir)
    test_df=pd.read_csv(test_df_dir)
    raw_df=pd.concat([train_df,val_df,test_df],axis=0)
    # print(raw_df.shape)
    # print(raw_df.head(10))
    raw_dict={raw_df['filename'].iloc[i]:raw_df['label'].iloc[i] for i in range(raw_df.shape[0])}
    # print(raw_dict['n0153282900000005.jpg'])
    if not os.path.exists(new_data_dir):
        os.mkdir(new_data_dir)
        for img in os.listdir(data_dir):
            if img.find('.jpg')==-1:
                continue
            if not os.path.exists(os.path.join(new_data_dir,raw_dict[img])):
                os.mkdir(os.path.join(new_data_dir,raw_dict[img]))
            shutil.move(os.path.join(data_dir,img),os.path.join(new_data_dir,raw_dict[img]))
    train_dir='./train_img'
    val_dir='./val_img'
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
        for dir in os.listdir(new_data_dir):
            os.mkdir(os.path.join(val_dir,dir))
            random_file=random.sample(os.listdir(os.path.join(new_data_dir,dir)),60)
            for file in random_file:
                shutil.move(os.path.join(new_data_dir,dir,file),os.path.join(val_dir,dir))
        os.rename(new_data_dir,train_dir)
        
        
    # for dir in os.listdir(val_dir):
    #     random_file=random.sample(os.listdir(os.path.join(val_dir,dir)),40)
    #     for file in random_file:
    #         shutil.move(os.path.join(val_dir,dir,file),os.path.join(train_dir,dir))
    
    
    train_img_form=transforms.Compose([
        transforms.Resize(128),
        transforms.RandomResizedCrop(96,scale=(0.8,1.0)),
        # transform.Resize(72),
        # transform.CenterCrop(64),
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    val_img_form=transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    train_set=ImageFolder(train_dir,transform=train_img_form)
    val_set=ImageFolder(val_dir,transform=val_img_form)
    # print(train_set.total_len(),val_set.total_len())
    # train_set.__setattr__(shuffle=True,num_workers=4)
    # val_set.__setattr__(shuffle=True,num_workers=4)
    return train_set,val_set

if __name__=='__main__':
    build_dataset()
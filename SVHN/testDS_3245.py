# Copyright 2020 Max Planck Institute for Software Systems

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from LazierGreedy import *
from pickle import dump,load
import sys
from sys import argv
from datetime import datetime
from scipy.io import loadmat
from tensorflow import keras
import time
from os import mkdir
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
os.environ["CUDA_VISIBLE_DEVICES"]="3" 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from os.path import exists
if not exists("DSBatched_26Nov_RQ3"):
    mkdir("DSBatched_26Nov_RQ3")
path="DSBatched_26Nov_RQ3/"+str(datetime.now())+"/"
mkdir(path)
sys.stdout=open(path+"log.txt","w")
loss=False
grs=4
# if len(argv)>2:
#     loss=argv[2]=="xent"
# if len(argv)>3:
#     grs=int(argv[3])
Data={}
succ=0
tot=0
# if argv[1]=="undef":
#     from madryCifarUndefWrapper import *
#     target_set=load(open("indices.pkl","rb"))
# else:
#     from madryCifarWrapper import *
#     target_set=load(open("def_indices.pkl","rb"))
#from madryCifarUndefWrapper import *
target_set=load(open("indices_12000_SVHN_share_robot_ds/3245_indexes_random_unique.pkl","rb")) #index to 3245 x_train
model = keras.models.load_model("./saved_model/Lenet5_svhn.h5")

data_train=loadmat("SVHN_dataset/train_32x32.mat")      
x_train=data_train['X']
y_train=data_train['y']%10  #10 will become 0
x_train=x_train.transpose(3,0,1,2)
x_train=x_train/255.0
# y_train=keras.utils.to_categorical(y_train, 10)

x_vali=x_train[60000:]
x_train=x_train[:60000]
y_vali=y_train[60000:]
y_train=y_train[:60000]

data_test=loadmat("SVHN_dataset/test_32x32.mat")
x_test=data_test['X']
y_test=data_test['y']%10
x_test=x_test.transpose(3,0,1,2)
x_test=x_test/255.0
# y_test=keras.utils.to_categorical(y_test, 10)


print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

print(f"x_vali shape: {x_vali.shape}")
print(f"y_vali shape: {y_vali.shape}")

print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

start_t = time.time()
for j in target_set:
    time_ran=time.time()-start_t
#     if time_ran>1200:
#         print(f"Actual Time ran is: {time_ran}")
#         break
    tot+=1
    print("Starting attack on image", tot, "with index",j)
    ret=DeepSearchBatched(x_train[j:j+1],model,y_train[j],8/255,max_calls=20000, batch_size=64,x_ent=loss,gr_init=grs)
    dump(ret[1].reshape(1,32,32,3),open(path+"image_"+str(j)+".pkl","wb"))
    Data[j]=(ret[0],ret[2])
    if ret[0]:
        succ+=1
        print("Attack Succeeded with",ret[2],"queries, success rate is",100*succ/tot)
    else:
        print("Attack Failed using",ret[2],"queries, success rate is",100*succ/tot)
    dump(Data,open(path+"data.pkl","wb"))
end_t = time.time()

time_taken=end_t-start_t
#dump(delta_t,open("test_time.pkl","wb"))
dump(time_taken,open(path+"time_taken_3245_indexes_svhn_26Nov_1412pm.pkl","wb"))
print(f"target_set.shape(should be 3245): {target_set.shape}")
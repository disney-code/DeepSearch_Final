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
import numpy as np 
from itertools import product
import heapq
from pickle import load
class Image:
    def __init__(self, base, model, true_class, epsilon, group_size=1, group_axes=[1,2], u_bound=1,l_bound=0,start_mode=-1, logits=False, x_ent=False,verbose=True):
        preprocess=lambda x:x.reshape(base.shape)
        self.orig_shape=base.shape
        self.calls=0
        self.group_axes=group_axes
        self.verbose=verbose
        self.group_size=group_size
        def predict(x):
            self.calls+=1
            return model.predict(preprocess(x)).reshape(-1)
        self.predict=predict
        self.true_class=true_class
        self.upper=np.clip(base.reshape(-1)+epsilon,l_bound,u_bound)
        self.lower=np.clip(base.reshape(-1)-epsilon,l_bound,u_bound)
        self.status=np.ones_like(self.lower)
        if start_mode>0:
            self.image=self.upper.copy()
        elif start_mode<0:
            self.image=self.lower.copy()
            self.status=self.status*-1
        else:
            self.image=base.reshape(-1)
            self.status=self.status*0
        self.gains=np.zeros_like(self.lower)
        self.gains_to=np.zeros_like(self.lower)
        def loss(image):
            #image shape is 3072,
            print(f"image shape: {image.shape}")
            res=self.predict(image)
            print(f"res: {res}")
            print(f"true_class: {true_class}")
            print(f"np.argmax(res): {np.argmax(res)}")
            if np.argmax(res)!=true_class:
                return -50000
            if logits:
                if x_ent:
                    return res[true_class]-np.log(np.sum(np.exp(res)))
                else:
                    rest=np.ones_like(res)
                    rest[true_class]=0
                    return res.true_class-np.max(res[rest>0])
            else:
                if x_ent:
                    return np.log(res[true_class])
                else:
                    rest=np.ones_like(res)
                    rest[true_class]=0
                    return np.log(res[true_class])-np.log(np.max(res[rest>0]))
        self.loss_fn=loss
        self.loss=loss(self.image)
        self.stale=np.zeros_like(self.gains)
        self.rmap=preprocess(np.arange(len(self.image)))
    def get_indices(self,source):
        j=[]
        for i in reversed(self.orig_shape):
            j.append(source % i)
            source=source//i
        j=list(reversed(j))
        for i in self.group_axes:
            j[i]=j[i]-(j[i] % self.group_size)
        indices=[]
        for i in range(len(self.orig_shape)):
            if i in self.group_axes:
                indices.append(list(range(j[i],j[i]+self.group_size)))
            else:
                indices.append([j[i]])
        ret=[]
        for k in product(*indices):
            ret.append(self.rmap[k])
        return ret
    def get_pivots(self,direction):
        indices=[]
        ret=[]
        for i in range(len(self.orig_shape)):
            indices.append(list(range(0,self.orig_shape[i],self.group_size if i in self.group_axes else 1)))
        for k in product(*indices):
            if direction==0 or self.status[self.rmap[k]]==direction:
                ret.append(self.rmap[k])
        return ret
    def gain(self, index, force=False, no_update=False, direction=0):
        if direction==0:
            if self.status[index]>0:
                direction=-1
            else:
                direction=1
        if self.gains_to[index]==direction and (not force):
            return self.gains[index]
        pert=self.image.copy()
        pert[self.get_indices(index)]=self.lower[self.get_indices(index)] if direction<0 else self.upper[self.get_indices(index)]
        res=self.loss_fn(pert)
        self.gains_to[index]=direction
        self.gains[index]=res-self.loss
        self.stale[index]=0
        return res-self.loss
    
    def push(self, index, loss_diff, direction=0):
        if direction==0:
            if self.status[index]>0:
                direction=-1
            else:
                direction=1
        self.image[self.get_indices(index)]=self.lower[self.get_indices(index)] if direction<0 else self.upper[self.get_indices(index)]
        self.status[self.get_indices(index)]=direction
        self.stale+=1
        self.loss+=loss_diff
        if self.verbose:
            print("Pushing group of",self.group_size,"beginning at",index,"to",["lower bound,","","upper bound,"][direction+1],"Current loss is",self.loss,"and",self.calls,"calls have been made to the model.")
    def reset(self):
        self.stale=self.stale*0
        self.gains_to=self.gains_to*0
        self.loss=self.loss_fn(self.image)
        if self.verbose:
            print("Purging gains cache")
    
    def sample_indices(self, count, direction=0):
        lst=self.get_pivots(direction)
        try:
            return np.random.choice(lst,count,False)
        except ValueError:
            return lst
    


def DeepSearchBatched(image,model,true_class,epsilon,max_calls,batch_size=64,randomize=True,x_ent=False,gr_init=4):
    target=Image(image,model,true_class,epsilon,group_size=gr_init,x_ent=x_ent)
    print("Initial loss is",target.loss)
    while(target.loss>-10000 and target.calls<max_calls):
        selected=[]
        cur_batch=0
        all_pivots=target.get_pivots(0)
        if randomize:
            np.random.shuffle(all_pivots)
        for x in all_pivots:
            cur_batch+=1
            if target.gain(x,True)<0:
                selected.append(x)
            if cur_batch==batch_size:
                for x in selected:
                    target.push(x,0)
                target.loss=target.loss_fn(target.image)
                if target.loss<-10000:
                    return True,target.image,target.calls
                if target.calls>max_calls:
                    return False,image,target.calls
                cur_batch=0
                selected=[]
        for x in selected:
            target.push(x,0)
        target.loss=target.loss_fn(target.image)
        if target.group_size>1:
            target.group_size=target.group_size//2
        target.reset()
    if target.loss>-10000:
        return False,image,target.calls
    return True,target.image,target.calls



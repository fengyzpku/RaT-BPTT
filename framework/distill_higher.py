import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.multiprocessing as mp
import higher

import numpy as np
import random

import time


from framework.config2 import get_arch

#@torch.no_grad()
def _weights_init(m):
    #if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        init.kaiming_normal_(m.weight)
        #m.reset_parameters()
#        try:
#            print(m.update)
#            print()


#    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        

        
class Distill(nn.Module):
    def __init__(self, x_init, y_init, arch, window, lr, num_train_eval, img_pc, batch_pc, num_classes=2, task_sampler_nc=2, train_y=False, train_lr=False, 
                 channel=3, im_size=(32, 32), inner_optim='SGD', syn_intervention=None, real_intervention=None, cctype=0, 
                 mixup=False, mix_prob=0, decay=False, inner_init=0, mse_loss=False, downsample_scale=1):
        super(Distill, self).__init__()
        #self.data = torch.nn.Parameter(x_init)
        self.data = nn.Embedding(img_pc*num_classes, int(channel*np.prod(im_size)/downsample_scale/downsample_scale))
        self.train_y = train_y
        self.downsample_scale = downsample_scale
        if train_y:
            self.label = nn.Embedding(img_pc*num_classes, num_classes)
            self.label.weight.data = y_init.float().cuda()
        else:
            self.label = y_init
        self.num_classes = num_classes
        self.channel = channel
        self.im_size = im_size
        self.net = get_arch(arch, self.num_classes, self.channel, self.im_size)
        self.img_pc = img_pc
        self.batch_pc = batch_pc
        self.arch = arch
        self.lr = lr if not train_lr else torch.nn.Parameter(lr)
        self.window = window
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        if mse_loss:
            self.mse_loss = True
            self.criterion = nn.MSELoss(reduction='mean')
        self.num_train_eval = num_train_eval
        self.curriculum = window
        self.inner_optim = inner_optim
        self.batch_id = 0
        self.boost_batch_id = 0
        self.syn_intervention = syn_intervention
        self.real_intervention = real_intervention
        self.cctype = cctype
        self.mixup_data = mixup
        self.mix_prob = mix_prob
        self.decay = decay
        #self.order_list = torch.randperm(self.img_pc)
        self.boost_ipc = None
        self.task_sampler_nc = task_sampler_nc
        #self.update_num = int(self.img_pc / self.batch_pc) if self.img_pc%self.batch_pc == 0 else int(2 * self.img_pc / self.batch_pc)
        #self.update_num = max(self.update_num, 1)
        #self.inner_init = inner_init
        #print(self.task_sampler_nc)
        
    def shuffle(self):
        #True
        self.order_list = torch.randperm(self.img_pc)
        if self.img_pc >= self.batch_pc:
            self.order_list = torch.cat([self.order_list, self.order_list], dim=0)
            
    def get_task_indices(self):
        task_indices = list(range(self.num_classes))
        if self.task_sampler_nc < self.num_classes:
            random.shuffle(task_indices)
            task_indices = task_indices[:self.task_sampler_nc]
            task_indices.sort()
        return task_indices
    
    def lim_shuffle(self):
        self.boost_list = torch.randperm(self.boost_ipc)
        if self.boost_ipc >= self.batch_pc:
            self.boost_list = torch.cat([self.boost_list, self.boost_list], dim=0)
        
        
    def subsample(self):
        indices = []
        if self.task_sampler_nc == self.num_classes:
            for i in range(self.num_classes):
                ind = torch.randperm(self.img_pc)[:self.batch_pc].sort()[0] + self.img_pc * i
                indices.append(ind)
        else:
            task_indices = self.get_task_indices()
            #print(task_indices)
            for i in task_indices:
                ind = torch.randperm(self.img_pc)[:self.batch_pc].sort()[0] + self.img_pc * i
                indices.append(ind)
        indices = torch.cat(indices).cuda()
        imgs    = self.data(indices)
        imgs = imgs.view(
                   self.task_sampler_nc * min(self.img_pc, self.batch_pc),
                   self.channel,
                   int(self.im_size[0]/self.downsample_scale),
                   int(self.im_size[1]/self.downsample_scale)
               ).contiguous()
        
        if self.downsample_scale > 1:
            imgs = F.interpolate(
                              imgs,
                              scale_factor=self.downsample_scale,
                              mode='bilinear',
                              align_corners=False
                          )
            
        if self.train_y:
            labels    = self.label(indices)
            labels = labels.view(
                       self.task_sampler_nc * min(self.img_pc, self.batch_pc),
                       self.num_classes
                   ).contiguous()
        else:
            labels = self.label[indices]
        
        #self.batch_id = (self.batch_id + 1) % self.update_num
        return imgs, labels
    
    def lim_subsample(self):
        indices = []
        for i in range(self.num_classes):
            ind = self.boost_list[self.boost_batch_id*self.batch_pc:(self.boost_batch_id+1)*self.batch_pc].sort()[0] + self.img_pc * i
            indices.append(ind)
        indices = torch.cat(indices).cuda()
        #if epoch <= 5:
        #    print(indices)
        imgs    = self.data(indices)
        imgs = imgs.view(
                   self.num_classes * min(self.boost_ipc, self.batch_pc),
                   self.channel,
                   self.im_size[0],
                   self.im_size[1]
               ).contiguous()
        labels = self.label[indices]
        
        self.boost_batch_id = (self.boost_batch_id + 1) % self.boost_update_num
        return imgs, labels


    def mixup(self, x, label):
        r = np.random.rand(1)
        label_b = label
        ratio = 0
        if r < self.mix_prob:
            # generate mixed sample
            lam = np.random.beta(1, 1)
            rand_index = random_indices(label, nclass=self.num_classes)

            label_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, label_b, ratio


    def forward(self, x):
        self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
        self.net.train()
        if self.boost_ipc is not None:
            
            if self.inner_optim == 'SGD':
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200], gamma=0.2) if self.decay else None
            elif self.inner_optim == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            self.lim_shuffle()
            for i in range(self.inner_init):
                if i == 1: print('performing inner startup')
                self.optimizer.zero_grad()
                imgs, label = self.lim_subsample()
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                ratio = 0
                if self.mixup_data: 
                    imgs, label_b, ratio = self.mixup(imgs, label)
                out, pres = self.net(imgs)
            
                if self.mixup_data:
                    loss = self.criterion(out, label) * ratio + self.criterion(out, label_b) * (1. - ratio)
                else:
                    loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                if self.inner_optim == 'SGD' and self.scheduler is not None:
                    self.scheduler.step()
            
        if self.inner_optim == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200], gamma=0.2) if self.decay else None
        elif self.inner_optim == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        if self.dd_type not in ['curriculum', 'standard']:
            print('The dataset distillation method is not implemented!')
            raise NotImplementedError()
        #self.shuffle()
        
        if self.dd_type == 'curriculum':
            for i in range(self.curriculum):
                self.optimizer.zero_grad()
                imgs, label = self.subsample()
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                ratio = 0
                if self.mixup_data: 
                    imgs, label_b, ratio = self.mixup(imgs, label)
                #print(imgs.shape)
                out, pres = self.net(imgs)
            
                if self.mixup_data:
                    loss = self.criterion(out, label) * ratio + self.criterion(out, label_b) * (1. - ratio)
                else:
                    loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                if self.inner_optim == 'SGD' and self.scheduler is not None:
                    self.scheduler.step()
                #self.shuffle()
        loss_coef = 1
        with higher.innerloop_ctx(
                self.net, self.optimizer, copy_initial_weights=True
            ) as (fnet, diffopt):
            for i in range(self.window):
                imgs, label = self.subsample()
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                ratio = 0
                if i + self.curriculum == 150 or i + self.curriculum == 240:
                    if self.inner_optim == 'SGD':
                        loss_coef = loss_coef * 0.2
                if self.mixup_data: imgs, label_b, ratio = self.mixup(imgs, label)
                out, pres = fnet(imgs)
            
                if self.mixup_data:
                    loss = self.criterion(out, label) * ratio + self.criterion(out, label_b) * (1. - ratio)
                else:
                    loss = self.criterion(out, label)
                diffopt.step(loss)
                if self.inner_optim == 'SGD' and self.scheduler is not None:
                    self.scheduler.step()
                #self.shuffle()
            x = self.real_intervention(x, dtype='real', seed=random.randint(0, 10000))
            return fnet(x)
        
    def continue_forward(self, x):
        loss_coef = 1
        with higher.innerloop_ctx(
                self.net, self.optimizer, copy_initial_weights=True
            ) as (fnet, diffopt):
            for i in range(self.window):
                imgs, label = self.subsample()
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                ratio = 0
                if i + self.curriculum == 150 or i + self.curriculum == 240:
                    if self.inner_optim == 'SGD':
                        loss_coef = loss_coef * 0.2
                if self.mixup_data: imgs, label_b, ratio = self.mixup(imgs, label)
                out, pres = fnet(imgs)
            
                if self.mixup_data:
                    loss = self.criterion(out, label) * ratio + self.criterion(out, label_b) * (1. - ratio)
                else:
                    loss = self.criterion(out, label)
                diffopt.step(loss)
                if self.inner_optim == 'SGD' and self.scheduler is not None:
                    self.scheduler.step()
                #self.shuffle()
            x = self.real_intervention(x, dtype='real', seed=random.randint(0, 10000))
            return fnet(x)

    def init_train(self, epoch, init=False, lim=True):
        if init:
            self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
            if self.boost_ipc is not None and lim:
                if self.inner_optim == 'SGD':
                    self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
                    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200], gamma=0.2) if self.decay else None
                elif self.inner_optim == 'Adam':
                    self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
                self.lim_shuffle()
                
                for i in range(self.inner_init):
                    if i == 1: print("start limiting training with IPC boost")
                    self.optimizer.zero_grad()
                    imgs, label = self.lim_subsample()
                    imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                    ratio = 0
                    if self.mixup_data: 
                        imgs, label_b, ratio = self.mixup(imgs, label)
                    out, pres = self.net(imgs)

                    if self.mixup_data:
                        loss = self.criterion(out, label) * ratio + self.criterion(out, label_b) * (1. - ratio)
                    else:
                        loss = self.criterion(out, label)
                    loss.backward()
                    self.optimizer.step()
                    if self.inner_optim == 'SGD' and self.scheduler is not None:
                        self.scheduler.step()
                    
            if self.inner_optim == 'SGD':
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[600], gamma=0.2) if self.decay else None
            elif self.inner_optim == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            #self.shuffle()
        for i in range(epoch):
            self.optimizer.zero_grad()
            imgs, label = self.subsample()
            imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
            out, pres = self.net(imgs)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            if self.inner_optim == 'SGD' and self.scheduler is not None:
                self.scheduler.step()
            #self.shuffle()
    
    # initialize the EMA with the currect data value
    def ema_init(self, ema_coef):
        self.shadow = -1e5
        #self.data.weight.clone()
        self.ema_coef = ema_coef
    
    # update the EMA value
    def ema_update(self, grad_norm):
        if self.shadow == -1e5: 
            self.shadow = grad_norm
        else:
            self.shadow -= (1 - self.ema_coef) * (self.shadow - grad_norm)
        return self.shadow
    
    # return the norm of EMA
    #def ema_norm(self):
    #    return torch.norm(self.shadow)
    #        
    
    def test(self, x):
        #print(x, x.dtype)
        #for name, param in self.net.named_parameters():
        #    print(name, param.data, param.dtype)
        with torch.no_grad():
            out = self.net(x)
        return out
    
        #grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
        

def random_indices(y, nclass=10, intraclass=False, device='cuda'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
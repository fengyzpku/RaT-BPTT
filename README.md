#   Correction of Figure 3

In the version initially submitted, **the line representing RaT-BPTT was inaccurate**. Instead of showcasing the intended RaT-BPTT, it displayed a continued training version of BPTT with 40 steps unrolling. The accurate representation of the RaT-BPTT curve can be found in the following graph.

<p align="middle">
<img src="gradients.png" alt="Meta-gradient norm" width="66%"/>
</p>

# RaT-BPTT
Random Truncated Backpropagation through Time for dataset distillation.

# Example Usage 

Before running the script, please install the environment in environment.yml. The key package here is the Higher package (https://github.com/facebookresearch/higher).

To distill on CIFAR-10 with 10 images per class:

`python main.py --dataset cifar10 --num_per_class 10 --batch_per_class 10 --num_train_eval 8 --world_size 1 --rank 0 --update_steps 1 --batch_size 5000 --ddtype curriculum --cctype 2 --epoch 60000 --test_freq 5 --print_freq 10 --arch convnet --init_type 1 --window 60 --minwindow 0 --totwindow 200 --inner_optim Adam --inner_lr 0.001 --lr 0.001 --zca --syn_strategy flip_rotate --real_strategy flip_rotate --fname 60_200 --data_sample none --seed 0`

In the above script, we use batch size 5000, window size 60, unroll length 200, and the Adam optimizer with 0.001 learning rate in both the inner loop and the outer loop. 

Due to space limits of the anonymous link, we are unable to upload the checkpoints for CIFAR100 IPC50 and for tiny-ImageNet. We will release these checkpoints later.
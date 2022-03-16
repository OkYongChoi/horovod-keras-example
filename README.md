# horovod (with Keras)
Modified the code from the NVIDA multi-GPU course

## Install horovod from dockerhub
I've tried other ways but failed to install horovod properly.
Thus, I strongly recommend you to pull the image of horovod from https://hub.docker.com/r/horovod/horovod.

You may need to pip install ```scipy``` and ```tensorflow-addons``` to run this script more on this docker image.


## Run the code
e.g. Single-Node 
```shell
$ horovodrun -np $num_gpus python wideresnet-cifar10.py --epochs 5 --batch-size 512
or
$ mpirun -np $num_gpus python wideresnet-cifar10.py --epochs 5 --batch-size 512
```

e.g. Multi-Node
$ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python wideresnet-cifar10.py
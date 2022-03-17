
I used WideResNet model for Cifar10 classifier and [NovoGrad](https://arxiv.org/abs/1905.11286) for optimizer

## install
```shell
$ pip install ray 
$ pip install torch_optimizer # for NovoGrad

### Run the script
```
python wideresenet-cifar10.py
``````

## Reference
[Distributed PyTorch of Ray docs](https://docs.ray.io/en/latest/raysgd/raysgd_pytorch.html)


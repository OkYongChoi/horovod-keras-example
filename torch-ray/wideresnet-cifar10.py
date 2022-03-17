import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import torch_optimizer as optim

import ray
from ray.util.sgd.torch import TorchTrainer
from ray.util.sgd.torch import TrainingOperator

def cifar_creator(config):
    """Returns dataloaders to be used in `train` and `validate`."""
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_loader = DataLoader(CIFAR10(root="~/data", download=True, transform=tfms), batch_size=config["batch"])
    validation_loader = DataLoader(CIFAR10(root="~/data", download=True, transform=tfms), batch_size=config["batch"])
    return train_loader, validation_loader

def model_creator(config):
    # download the WRN from torch-hub
    return torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)

def optimizer_creator(model, config):
    """Returns an optimizer (or multiple)"""
    return optim.NovoGrad(model.parameters(), lr=config["lr"])

cifar10_TrainingOperator = TrainingOperator.from_creators(
    model_creator=model_creator, # A function that returns a nn.Module
    optimizer_creator=optimizer_creator, # A function that returns an optimizer
    data_creator=cifar_creator, # A function that returns dataloaders
    loss_creator=torch.nn.CrossEntropyLoss  # A loss function
    )

# For distributed Multi-node Training,
# ray.init(address="auto") 
# or a specific Ray address of the for "ip-address:port"
ray.init() 

# Under the hood, TorchTrainer will create 
# replicas of the model (controlled by num_workers),
# each of which is managed by a worker (Ray actor).
trainer = TorchTrainer(
    training_operator_cls=cifar10_TrainingOperator,
    config={"lr": 0.01, # used in optimizer_creator
            "batch": 256 # used in data_creator
           },
    num_workers=4,  # amount of parallelism
    use_fp16=True,  # Default is the native mixed precision training 

    use_gpu=torch.cuda.is_available(),
    use_tqdm=True)

stats = trainer.train()
print(trainer.validate())

torch.save(trainer.state_dict(), "checkpoint.pt")
trainer.shutdown()
print("success!")
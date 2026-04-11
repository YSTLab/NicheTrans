import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.data_loader import *


def sma_dataloader(args, dataset):
    transform_train = transforms.Compose([
            # transforms.RandomCrop([args.img_size, args.img_size]),
            
            transforms.RandomHorizontalFlip(0.5), 
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

            transforms.ToTensor(), 

            # transforms.Resize([args.img_size, args.img_size]),
            transforms.RandomResizedCrop(size=[args.img_size, args.img_size]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    transform_test = transforms.Compose([
            transforms.Resize([args.img_size, args.img_size]),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    pin_memory = torch.cuda.is_available()

    trainloader = DataLoader(
                    SMA_loader(dataset.training, transform=transform_train),
                    batch_size=args.train_batch,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=True,
                )

    testloader = DataLoader(
                    SMA_loader(dataset.testing, transform=transform_test),
                    batch_size=args.test_batch,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )
    return trainloader, testloader


def human_node_dataloader(args, dataset):
    pin_memory = torch.cuda.is_available()
    trainloader = DataLoader(
                    Lymph_node_loader(dataset.training),
                    batch_size=args.train_batch,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=True,
                )

    testloader = DataLoader(
                    Lymph_node_loader(dataset.testing),
                    batch_size=args.test_batch,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )
    return trainloader, testloader

def embryonic_mouse_brain(args, dataset):
    pin_memory = torch.cuda.is_available()
    trainloader = DataLoader(
                    Lymph_node_loader(dataset.training),
                    batch_size=args.train_batch,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=True,
                )

    testloader = DataLoader(
                    Lymph_node_loader(dataset.testing),
                    batch_size=args.test_batch,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )
    return trainloader, testloader

def breast_cancer_dataloader(args, dataset):
    pin_memory = torch.cuda.is_available()
    trainloader = DataLoader(
                    Breast_cancer_loader(dataset.training),
                    batch_size=args.train_batch,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=True,
                )

    testloader = DataLoader(
                    Breast_cancer_loader(dataset.testing),
                    batch_size=args.test_batch,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )
    return trainloader, testloader

def ad_mouse_dataloader(args, dataset, testing_control=False):
    pin_memory = torch.cuda.is_available()
    trainloader = DataLoader(
                    AD_Mouse_loader(dataset.training),
                    batch_size=args.train_batch,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=True,
                )

    testloader = DataLoader(
                    AD_Mouse_loader(dataset.testing),
                    batch_size=args.test_batch,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )

    valloader = DataLoader(
                    AD_Mouse_loader(dataset.val),
                    batch_size=args.test_batch,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )

    return trainloader, testloader, valloader
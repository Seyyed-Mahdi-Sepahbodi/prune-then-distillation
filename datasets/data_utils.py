import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datasets import custom_datasets   # ماژول دیتاست سفارشی (مثل Tiny-ImageNet)


def get_dataloader(dataset_type, batch_size, validation_size=0,
                   dataloader_seed=0, loader_type=None, length=None):
    """
    ساخت DataLoader برای دیتاست‌های مختلف.
    
    پارامترها:
        dataset_type (str): نوع دیتاست (cifar10, cifar100, tiny_imagenet, imagenet).
        batch_size (int): سایز بچ.
        validation_size (int یا float): اندازه‌ی مجموعه‌ی اعتبارسنجی (int=تعداد نمونه، float=درصد).
        dataloader_seed (int): بذر برای split تصادفی.
        loader_type (str): نوع بارگذاری:
            - 'train'     → فقط train
            - 'train_val' → train + validation
            - 'test'      → test/val
            - 'synflow'   → زیرمجموعه‌ی محدود از train برای محاسبه‌ی نمره‌ی SynFlow
        length (int): فقط برای synflow (تعداد نمونه‌های موردنیاز).
    
    خروجی:
        torch.utils.data.DataLoader یا tuple(train_loader, val_loader)
    """

    # تعیین حالت train/test
    if loader_type in ["train", "train_val", "synflow"]:
        train = True
    elif loader_type == "test":
        train = False
    else:
        raise ValueError("loader_type باید یکی از train, train_val, test, synflow باشد.")

    # مسیر ذخیره‌ی دیتاست‌ها
    save_path = f'../datasets/{dataset_type}'

    # ---- انتخاب دیتاست و ترنسفورم‌ها ----
    if dataset_type == 'cifar10':
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR10(save_path, train=train, download=True, transform=transform)

    elif dataset_type == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR100(save_path, train=train, download=True, transform=transform)

    elif dataset_type == 'tiny_imagenet':
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = get_transform(size=64, padding=4, mean=mean, std=std, preprocess=True)
        dataset = custom_datasets.TINYIMAGENET('../datasets', train=train,
                                               download=True, transform=transform)

    elif dataset_type == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            # ترنسفورم‌های augment برای آموزش
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            folder = os.path.join(save_path, 'train')
        else:
            # ترنسفورم‌های استاندارد برای تست/val
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            folder = os.path.join(save_path, 'val')
        dataset = datasets.ImageFolder(folder, transform=transform)

    else:
        raise ValueError(f"دیتاست ناشناخته: {dataset_type}")

    # ---- ساخت DataLoader بر اساس نوع ----
    kwargs = {}

    if loader_type == 'train_val':
        # محاسبه‌ی تعداد نمونه‌ی val
        if 2 <= validation_size < len(dataset):
            split_line = validation_size
        elif 0 <= validation_size <= 1:
            split_line = round(len(dataset) * validation_size)
        else:
            raise ValueError(f"validation_size نامعتبر ({validation_size})")

        # تقسیم dataset به train/val
        trainset, valset = torch.utils.data.random_split(
            dataset,
            [len(dataset) - split_line, split_line],
            generator=torch.Generator().manual_seed(dataloader_seed)
        )

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                 shuffle=False, **kwargs)
        return train_loader, val_loader

    elif loader_type == 'train':
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, **kwargs)

    elif loader_type == "synflow":
        if length is None:
            raise ValueError("برای synflow باید پارامتر length تعیین شود.")
        indices = torch.randperm(len(dataset))[:length]
        subset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                           shuffle=True, **kwargs)

    elif loader_type == "test":
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, **kwargs)


def device(gpu: int):
    """بررسی CUDA و انتخاب GPU (در غیر این صورت CPU)."""
    use_cuda = torch.cuda.is_available()
    return torch.device(f"cuda:{gpu}" if use_cuda else "cpu")


def get_transform(size, padding, mean, std, preprocess):
    """
    ساخت ترنسفورم استاندارد برای داده‌ها.
    - اگر preprocess=True → شامل Crop و Flip
    - همیشه → ToTensor + Normalize
    """
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)

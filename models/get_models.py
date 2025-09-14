# انتخاب و ساخت مدل بر اساس نوع مدل و دیتاست

from models import cifar_vgg, tinyimagenet_resnet, tinyimagenet_vgg, tinyimagenet_mobilenetv2


def get_model(model_type, dataset, plan=None):
    """
    دریافت سازنده‌ی مدل بر اساس نام مدل و دیتاست و سپس ساخت نمونه‌ی مدل.

    ورودی‌ها:
        model_type (str): نام/نوع مدل (مثلاً 'vgg19', 'resnet18', 'vgg-custom', ...)
        dataset (str): نام دیتاست برای تعیین input_shape و num_classes.
        plan (list|None): فقط برای مدل‌های نوع 'custom' (مثلاً vgg-custom).

    خروجی:
        model (torch.nn.Module): نمونه‌ی مدل ساخته‌شده.
    """

    # تعیین شکل ورودی و تعداد کلاس‌ها بر اساس دیتاست
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    elif dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    elif dataset == 'tiny_imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    elif dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000

    # فهرست مدل‌های پشتیبانی‌شده برای CIFAR-100
    cifar100_models = {
        'vgg11': cifar_vgg.vgg11_bn,
        'vgg19': cifar_vgg.vgg19_bn,
        'vgg19-rwd-cl1': cifar_vgg.vgg19_rwd_cl1,
        'vgg19-rwd-cl2': cifar_vgg.vgg19_rwd_cl2,
        'vgg19-rwd-st36': cifar_vgg.vgg19_rwd_st36,
        'vgg19-rwd-st59': cifar_vgg.vgg19_rwd_st59,
        'vgg19-rwd-st79': cifar_vgg.vgg19_rwd_st79,
        'vgg19dbl': cifar_vgg.vgg19dbl,
        'vgg19dbl-rwd-st36': cifar_vgg.vgg19dbl_rwd_st36,
        'vgg19dbl-rwd-st59': cifar_vgg.vgg19dbl_rwd_st59,
        'vgg19dbl-rwd-st79': cifar_vgg.vgg19dbl_rwd_st79,
        'vgg-custom': cifar_vgg.vgg_custom
    }

    # فهرست مدل‌های پشتیبانی‌شده برای Tiny-ImageNet
    tiny_imagenet_models = {
        'vgg16': tinyimagenet_vgg.vgg16_bn,
        'resnet18': tinyimagenet_resnet.resnet18,
        'resnet18-rwd-st36': tinyimagenet_resnet.resnet18_rwd_st36,
        'resnet18-rwd-st59': tinyimagenet_resnet.resnet18_rwd_st59,
        'resnet18-rwd-st79': tinyimagenet_resnet.resnet18_rwd_st79,
        'resnet18dbl': tinyimagenet_resnet.resnet18dbl,
        'resnet18dbl-rwd-st36': tinyimagenet_resnet.resnet18dbl_rwd_sp36,
        'resnet18dbl-rwd-st59': tinyimagenet_resnet.resnet18dbl_rwd_sp59,
        'resnet18dbl-rwd-st79': tinyimagenet_resnet.resnet18dbl_rwd_sp79,
        'resnet50': tinyimagenet_resnet.resnet50,
        'resnet50-rwd-sp36': tinyimagenet_resnet.resnet50_rwd_sp36,
        'resnet50-rwd-sp59': tinyimagenet_resnet.resnet50_rwd_sp59,
        'resnet50-rwd-sp79': tinyimagenet_resnet.resnet50_rwd_sp79,
        'mobilenet-v2': tinyimagenet_mobilenetv2._mobilenet_v2,
    }

    # نگاشت دیتاست → فهرست مدل‌های مجاز
    models = {
        'cifar100': cifar100_models,
        'tiny_imagenet': tiny_imagenet_models
    }

    # بررسی پشتیبانی دیتاست/مدل
    if dataset not in models:
        raise ValueError(f"{dataset} is not supported.")
    if model_type not in models[dataset]:
        raise ValueError(
            f"{model_type} is not supported in {dataset}. Check the dataset or model_type.\n"
            f"Supported model in {dataset}:\n"
            f"{list(models[dataset].keys())}"
        )

    # ساخت مدل (برای مدل‌های custom، plan نیز پاس داده می‌شود)
    if "custom" in model_type:
        model = models[dataset][model_type](input_shape, num_classes, plan)
    else:
        model = models[dataset][model_type](input_shape, num_classes)

    # برچسب نوع مدل برای استفاده‌های بعدی
    model.model_type = model_type

    return model

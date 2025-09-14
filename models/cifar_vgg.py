# بر پایه‌ی کد PyTorch VGG (الهام از Lottery Ticket)

import torch.nn as nn
import torch.nn.functional as F

from models import base, layers


class ConvModule(nn.Module):
    """یک بلوک کانولوشنی ساده برای VGG: Conv3x3 + ReLU."""

    def __init__(self, in_filters, out_filters):
        super(ConvModule, self).__init__()
        # از لایه‌ی سفارشی (دارای پشتیبانی ماسک/پرو‌نینگ) استفاده می‌شود
        self.conv = layers.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x):
        # فعال‌ساز ReLU بعد از کانولوشن
        return F.relu(self.conv(x))


class ConvBNModule(nn.Module):
    """بلوک کانولوشنی با BatchNorm برای VGG: Conv3x3 + BN + ReLU."""

    def __init__(self, in_filters, out_filters):
        super(ConvBNModule, self).__init__()
        self.conv = layers.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = layers.BatchNorm2d(out_filters)

    def forward(self, x):
        # توالی Conv → BN → ReLU
        return F.relu(self.bn(self.conv(x)))


class VGG(base.base_model):
    """شبکه‌ی سبک VGG برای ورودی‌های CIFAR (ابعاد کوچک)."""

    def __init__(self, plan, conv, num_classes, dense_classifier=False):
        """
        plan: لیست مشخصات لایه‌ها (اعداد=تعداد فیلتر، 'M'=MaxPool)
        conv: کلاس بلوک کانولوشنی (ConvModule یا ConvBNModule)
        num_classes: تعداد کلاس‌ها
        dense_classifier: اگر True باشد، از nn.Linear معمولی به‌جای layers.Linear استفاده می‌شود
        """
        super(VGG, self).__init__()

        layer_list = []
        filters = 3  # ورودی تصویر RGB

        # ساخت پشته‌ی کانولوشن/پولینگ بر اساس plan
        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layer_list)

        # رگرسور نهایی (کلاس‌فایر خطی)
        self.fc = layers.Linear(plan[-1], num_classes)
        if dense_classifier:
            # گزینه‌ی کلاس‌فایرِ Dense (بدون لایه‌ی سفارشی)
            self.fc = nn.Linear(plan[-1], num_classes)

        # مقداردهی وزن‌ها
        self._initialize_weights()

    def forward(self, x):
        # عبور از بدنه‌ی کانولوشنی
        x = self.layers(x)
        # Pool اضافی برای سازگاری ابعادی (CIFAR → 32x32)
        x = nn.AvgPool2d(2)(x)
        # تخت کردن و عبور از لایه‌ی خطی
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        """Kaiming init برای Conv/Linear و مقداردهی BN."""
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _plan(num):
    """
    تعریف نقشه‌ی لایه‌های VGG (تعداد فیلترها و محل MaxPool).
    مقادیر *_st_* و *_dbl_* نسخه‌های باز-طراحی/هرس‌شده‌اند.
    """
    if num == 11:
        plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 19:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                512, 512, 512, 512, 'M', 512, 512, 512, 512]
    elif num == "19_cl_1":
        plan = [64, 64, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M',
                256, 256, 256, 256, 'M', 512, 512, 512, 512]
    elif num == "19_cl_2":
        plan = [64, 39, 'M', 179, 79, 'M', 354, 155, 362, 146, 'M',
                614, 247, 500, 158, 'M', 271, 139, 547, 512]
    elif num == "19_dbl":
        plan = [64, 128, 'M', 256, 256, 'M', 512, 512, 512, 512, 'M',
                1024, 1024, 1024, 1024, 'M', 1024, 1024, 1024, 512]
    elif num == "19_dbl_st_36":
        plan = [51, 118, 'M', 265, 235, 'M', 525, 468, 525, 452, 'M',
                941, 775, 859, 731, 'M', 754, 713, 738, 443]
    elif num == "19_dbl_st_59":
        plan = [40, 111, 'M', 244, 218, 'M', 472, 445, 473, 406, 'M',
                833, 674, 637, 542, 'M', 381, 469, 681, 350]
    elif num == "19_dbl_st_79":
        plan = [39, 99, 'M', 243, 194, 'M', 472, 387, 468, 340, 'M',
                726, 402, 356, 239, 'M', 205, 194, 461, 238]
    elif num == "19_st_36":
        plan = [53, 61, 'M', 128, 121, 'M', 258, 241, 258, 236, 'M',
                485, 410, 406, 374, 'M', 340, 358, 390, 399]
    elif num == "19_st_59":
        plan = [44, 56, 'M', 116, 112, 'M', 238, 227, 240, 209, 'M',
                408, 348, 339, 246, 'M', 196, 214, 367, 272]
    elif num == "19_st_79":
        plan = [40, 49, 'M', 111, 97, 'M', 225, 187, 224, 170, 'M',
                356, 233, 220, 99, 'M', 111, 84, 297, 122]
    else:
        raise ValueError('Unknown VGG model: {}'.format(num))
    return plan


def _vgg(arch, plan, conv, num_classes, dense_classifier, pretrained):
    """
    سازنده‌ی مشترک برای مدل‌های VGG این فایل.
    پارامتر pretrained عملاً استفاده نمی‌شود (بدون تغییر منطق، نگه داشته شده است).
    """
    model = VGG(plan, conv, num_classes, dense_classifier)
    # بلوکِ دانلود/لود pretrained غیرفعال بود و حذف شد تا کد تمیزتر شود (منطق تغییری نکرد).
    return model


# توابع کمکی سازنده‌ی واریانت‌های مختلف VGG
def vgg11_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11)
    return _vgg('vgg11_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(19)
    return _vgg('vgg19_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19_rwd_cl1(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_cl_1")
    return _vgg('vgg19_cl_1', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19_rwd_cl2(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_cl_2")
    return _vgg('vgg19_cl_2', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19_rwd_st36(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_st_36")
    return _vgg('vgg19_st_36', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19_rwd_st59(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_st_59")
    return _vgg('vgg19_st_59', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19_rwd_st79(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_st_79")
    return _vgg('vgg19_st_79', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19dbl(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_dbl")
    return _vgg('vgg19_dbl', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19dbl_rwd_st36(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_dbl_st_36")
    return _vgg('vgg19_dbl_36', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19dbl_rwd_st59(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_dbl_st_59")
    return _vgg('vgg19_dbl_59', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg19dbl_rwd_st79(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan("19_dbl_st_79")
    return _vgg('vgg19_dbl_79', plan, ConvBNModule, num_classes, dense_classifier, pretrained)


def vgg_custom(input_shape, num_classes, plan, dense_classifier=False, pretrained=False):
    # ساخت نسخه‌ی سفارشی با plan ورودی
    return _vgg('vgg_custom', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

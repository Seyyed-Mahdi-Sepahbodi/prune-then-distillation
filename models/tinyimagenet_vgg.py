# پیاده‌سازی VGG برای Tiny-ImageNet با لایه‌های دارای ماسک (layers.*)
import torch
import torch.nn as nn
from models import base, layers

# پیکربندی‌های استاندارد VGG (A/B/D/E). در این فایل از 'D' برای VGG16-BN استفاده می‌شود.
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(base.base_model):
    """
    بدنه‌ی VGG:
      - features: بلوک‌های کانولوشن/BN/ReLU/MaxPool ساخته‌شده با make_layers
      - classifier: سه لایه‌ی خطی با Dropout و ReLU
      - num_classes پیش‌فرض 200 (مطابق Tiny-ImageNet)
    """

    def __init__(self, features, num_classes=200, dense_classifier=False):
        super().__init__()
        self.features = features

        # نوع لایه‌ی خطی (در این نسخه از layers.Linear استفاده می‌شود).
        # پارامتر dense_classifier به‌دلایل سازگاری امضای تابع حفظ شده است.
        self.Linear = layers.Linear

        # برای ورودی 64x64 و انتهای ویژگی 512ch، بعد از 5 بار MaxPool → 2x2 → 512*4
        dim = 512 * 4

        # کلاس‌فایر: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
        self.classifier = nn.Sequential(
            self.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.Linear(dim // 2, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.Linear(dim // 2, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)                 # استخراج ویژگی‌ها
        x = x.view(x.size(0), -1)            # تخت‌سازی
        x = self.classifier(x)               # طبقه‌بندی نهایی
        return x

    def _initialize_weights(self):
        """مقداردهی وزن‌ها مطابق عرف VGG."""
        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (layers.Linear, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg_list, batch_norm=False):
    """
    ساخت توالی لایه‌ها از روی لیست پیکربندی:
      - عدد: Conv3x3 (+BatchNorm اختیاری) + ReLU
      - 'M': MaxPool2d(2,2)
    """
    layer_list = []
    in_ch = 3
    for spec in cfg_list:
        if spec == 'M':
            layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        layer_list.append(layers.Conv2d(in_ch, spec, kernel_size=3, padding=1))
        if batch_norm:
            layer_list.append(layers.BatchNorm2d(spec))
        layer_list.append(nn.ReLU(inplace=True))
        in_ch = spec

    return nn.Sequential(*layer_list)


def _vgg(arch, features, num_classes, dense_classifier, pretrained):
    """
    سازنده‌ی مشترک برای مدل‌های VGG همین فایل.
    - در صورت pretrained=True، وزن‌ها از مسیر محلی بارگذاری می‌شوند (بدون تغییر منطق).
    """
    model = VGG(features, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-cifar{}.pt'.format(arch, num_classes)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg16_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """VGG-16 با BatchNorm مطابق پیکربندی 'D'."""
    features = make_layers(cfg['D'], batch_norm=True)
    return _vgg('vgg16_bn', features, num_classes, dense_classifier, pretrained)

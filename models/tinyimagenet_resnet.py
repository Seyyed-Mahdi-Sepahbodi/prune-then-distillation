# پیاده‌سازی ResNet (الهام از PyTorch CIFAR100 و Synaptic-Flow)
import torch
import torch.nn as nn
from models import base, layers


class BasicBlock(nn.Module):
    """بلوک پایه برای ResNet-18/34."""
    expansion = 1  # ضریب گسترش کانال‌ها در خروجی بلوک

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        # مسیر Residual: Conv3x3 → BN → ReLU → Conv3x3 → BN
        self.residual_function = nn.Sequential(
            layers.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layers.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            layers.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            layers.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # مسیر میان‌بُر (Shortcut): پیش‌فرض Identity (بدون تغییر ابعاد)
        self.shortcut = layers.Identity2d(in_channels)

        # اگر اندازه‌ی فضایی/کانالی تغییر کند، از 1x1 Conv برای هم‌تراز کردن استفاده می‌کنیم
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                layers.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                layers.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        # خروجی بلوک: ReLU(Residual + Shortcut)
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """بلوک Bottleneck برای ResNet-50/101/152."""
    expansion = 4  # خروجی بلوک 4 برابر out_channels است

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        width = int(out_channels * (base_width / 64.))

        # مسیر Residual: 1x1 → 3x3 → 1x1 (با BN و ReLU بینشان)
        self.residual_function = nn.Sequential(
            layers.Conv2d(in_channels, width, kernel_size=1, bias=False),
            layers.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            layers.Conv2d(width, width, stride=stride, kernel_size=3, padding=1, bias=False),
            layers.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            layers.Conv2d(width, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            layers.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        # مسیر Shortcut: پیش‌فرض Identity
        self.shortcut = layers.Identity2d(in_channels)

        # هم‌ترازی ابعاد در صورت تغییر اندازه‌ی فضایی/کانالی
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                layers.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                layers.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(base.base_model):
    """
    بدنه‌ی کلی ResNet برای Tiny-ImageNet (num_classes=200 به‌صورت پیش‌فرض).
    - conv1 با گزینه‌ی resopt تعیین می‌شود (بدون/با MaxPool اولیه).
    - چهار مرحله‌ی کانولوشنی (conv2_x تا conv5_x) براساس تعداد بلوک‌ها ساخته می‌شوند.
    """

    def __init__(self, block, num_block, base_width, num_classes=200, dense_classifier=False,
                 block_size=[64, 128, 256, 512, 512], resopt=True):
        super().__init__()

        self.in_channels = 64
        self.conv1 = self._res_optimizer(resopt)  # لایه‌ی ورود: Conv-BN-ReLU (با/بدون MaxPool)

        # ساخت مراحل ResNet؛ stride مرحله‌ی اول 1 است (به دلیل ورودی‌های کوچک‌تر از ImageNet)
        self.conv2_x = self._make_layer(block, block_size[0], num_block[0], 1, base_width)
        self.conv3_x = self._make_layer(block, block_size[1], num_block[1], 2, base_width)
        self.conv4_x = self._make_layer(block, block_size[2], num_block[2], 2, base_width)
        self.conv5_x = self._make_layer(block, block_size[3], num_block[3], 2, base_width)

        # Pool و کلاس‌فایر نهایی
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = layers.Linear(block_size[4] * block.expansion, num_classes)
        if dense_classifier:
            self.fc = nn.Linear(block_size[4] * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        # مقداردهی وزن‌ها برای Conv/BN (مطابق عرف ResNet)
        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layers.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, base_width):
        """
        ساخت یک مرحله از ResNet شامل چند بلوک پیاپی.
        - اولین بلوک با stride داده‌شده، بلوک‌های بعدی با stride=1
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for s in strides:
            layer_list.append(block(self.in_channels, out_channels, s, base_width))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layer_list)

    def _res_optimizer(self, resopt):
        """
        تعیین معماری conv1:
        - اگر resopt=True: Conv-BN-ReLU
        - اگر resopt=False: Conv-BN-ReLU + MaxPool (سبک نسخه‌ی ImageNet)
        """
        if resopt is True:
            return nn.Sequential(
                layers.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                layers.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                layers.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                layers.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x):
        # گذر رو به جلو: conv1 → conv2_x..conv5_x → GAP → FC
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _resnet(arch, block, num_block, base_width, num_classes, dense_classifier, pretrained, block_size, resopt):
    """
    سازنده‌ی مشترک برای همه‌ی واریانت‌های ResNet همین فایل.
    - در صورت pretrained=True، وزن‌های از مسیر محلی بارگذاری می‌شوند (بدون تغییر در منطق موجود).
    """
    model = ResNet(block, num_block, base_width, num_classes, dense_classifier, block_size, resopt)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-cifar{}.pt'.format(arch, num_classes)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


# === توابع سازنده‌ی مدل‌ها (Tiny-ImageNet: num_classes=200) ===

def resnet18(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-18 استاندارد (base_width=64)."""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[64, 128, 256, 512, 512], resopt=True)


def resnet18_rwd_st36(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-18 با کانال‌های بازطراحی‌شده (st_36)."""
    return _resnet('resnet18-st-1', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[58, 113, 216, 398, 398], resopt=True)


def resnet18_rwd_st59(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-18 با کانال‌های بازطراحی‌شده (st_59)."""
    return _resnet('resnet18-st-2', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[54, 102, 184, 305, 305], resopt=True)


def resnet18_rwd_st79(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-18 با کانال‌های بازطراحی‌شده (st_79)."""
    return _resnet('resnet18-st-3', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[49, 85, 140, 198, 198], resopt=True)


def resnet18dbl(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-18 با دوبرابر کردن کانال‌ها (dbl)."""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[128, 256, 512, 1024, 1024], resopt=True)


def resnet18dbl_rwd_sp36(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-18 دوبرابر شده با بازطراحی (sp_36)."""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[116, 224, 427, 796, 796], resopt=True)


def resnet18dbl_rwd_sp59(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-18 دوبرابر شده با بازطراحی (sp_59)."""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[104, 192, 353, 618, 618], resopt=True)


def resnet18dbl_rwd_sp79(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-18 دوبرابر شده با بازطراحی (sp_79)."""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[86, 148, 262, 432, 432], resopt=True)


def resnet50(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-50 (Bottleneck)."""
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained,
                   block_size=[64, 128, 256, 512, 512], resopt=True)


def resnet50_rwd_sp36(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-50 با بازطراحی کانال‌ها (sp_36)."""
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained,
                   block_size=[58, 110, 210, 402, 402], resopt=True)


def resnet50_rwd_sp59(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-50 با بازطراحی کانال‌ها (sp_59)."""
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained,
                   block_size=[48, 90, 164, 330, 330], resopt=True)


def resnet50_rwd_sp79(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ResNet-50 با بازطراحی کانال‌ها (sp_79)."""
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained,
                   block_size=[36, 60, 110, 246, 246], resopt=True)

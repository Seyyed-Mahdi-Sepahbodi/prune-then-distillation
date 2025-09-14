# پیاده‌سازی MobileNetV2 با لایه‌های دارای ماسک (layers.*) و کلاس پایه‌ی پروژه (base.base_model)
import torch
from torch import nn, Tensor
from typing import Callable, Any, Optional, List
from models import base, layers


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    تضمین می‌کند تعداد کانال‌ها مضربِ 'divisor' باشد (روش مرسوم در MobileNet).
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # جلوگیری از کاهش بیش از ۱۰٪ نسبت به مقدار اصلی
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    """
    بلوک: Conv2d (بدون بایاس) → Norm (پیش‌فرض BatchNorm2d سفارشی) → ReLU6
    از لایه‌های سفارشی پروژه (layers.*) استفاده می‌شود تا ماسک/پرونینگ پشتیبانی شود.
    """
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = layers.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super().__init__(
            layers.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                          dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# نام معادل برای سازگاری با کدهای قدیمی
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    """
    بلوک معکوس-رزیدوال MobileNetV2:
      - اختیاری: 1x1 pw expand
      - 3x3 dw (depthwise)
      - 1x1 pw-linear
      - اتصال میان‌بر وقتی stride=1 و ابعاد ورودی/خروجی برابر باشند
    """
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = layers.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1 and inp == oup)

        mbv2_layers: List[nn.Module] = []
        # مرحله‌ی expand (در صورت نیاز)
        if expand_ratio != 1:
            mbv2_layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        # dw + pw-linear
        mbv2_layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            layers.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*mbv2_layers)
        self.out_channels = oup
        self._is_cn = stride > 1  # پرچم داخلی برای برخی کاربردها

    def forward(self, x: Tensor) -> Tensor:
        # اگر ساختار اجازه دهد، اتصال میان‌بر اضافه می‌شود
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(base.base_model):
    """
    بدنه‌ی اصلی MobileNetV2.
    نکته: num_classes پیش‌فرض 200 (مطابق Tiny-ImageNet) تنظیم شده است.
    """
    def __init__(
        self,
        num_classes: int = 200,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = layers.BatchNorm2d

        # کانال‌های اولیه و نهایی طبق مقاله
        input_channel = 32
        last_channel = 1280

        # پیکربندی پیش‌فرض بلوک‌ها: [t, c, n, s]
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # بررسی صحت ساختار ورودی
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty or a 4-element list.")

        # لایه‌ی نخست: ConvBNReLU با stride=2
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]

        # ساخت بلوک‌های inverted residual
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        # لایه‌ی انتهایی قبل از کلاسیفایر: 1x1 ConvBNReLU
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))

        # تبدیل به Sequential
        self.features = nn.Sequential(*features)

        # کلاسیفایر: Dropout → Linear (نسخه‌ی سفارشی با ماسک)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            layers.Linear(self.last_channel, num_classes),
        )

        # مقداردهی وزن‌ها (مطابق نسخه‌ی رسمی)
        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (layers.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, layers.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        پیاده‌سازی forward (الگوی رسمی TorchVision برای سازگاری با TorchScript).
        """
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    سازنده‌ی MobileNetV2.
    پارامترهای pretrained/progress نگه داشته شده‌اند (بدون بارگذاری وزن از وب در این نسخه).
    """
    model = MobileNetV2(**kwargs)
    return model


def _mobilenet_v2(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """
    امضای سازگار با زیرساخت فراخوانی مدل‌ها در پروژه.
    (منطق اصلی تغییر نکرده است.)
    """
    return mobilenet_v2()

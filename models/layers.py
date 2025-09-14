# لایه‌های دارای «ماسک» برای پشتیبانی از Pruning بدون تغییر منطق محاسبات
# منبع ایده‌ی ماسک: SynFlow (https://github.com/ganguli-lab/Synaptic-Flow)
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class Linear(nn.Linear):
    """
    نسخه‌ی Linear با ماسک روی وزن/بایاس.
    - ماسک‌ها به‌صورت Buffer (روی دستگاه مدل) نگه‌داری می‌شوند.
    - در forward وزن/بایاس با ماسک ضرب می‌شوند تا وزن‌های هرس‌شده صفر بمانند.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
        # ماسک وزن و (در صورت وجود) ماسک بایاس
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return F.linear(input, W, b)


class Conv2d(nn.Conv2d):
    """
    نسخه‌ی Conv2d با ماسک روی وزن/بایاس.
    - سازوکار مشابه Linear: اعمال ماسک‌ها در مسیر forward.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode
        )
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        # همان پیاده‌سازی Conv2d استاندارد؛ تنها تفاوت این است که weight/bias
        # قبل از فراخوانی، با ماسک ضرب شده‌اند.
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return self._conv_forward(input, W, b)


class BatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d با ماسک روی پارامترهای affine (در صورت فعال بودن).
    - اگر affine=False باشد، مانند BatchNorm استاندارد عمل می‌کند.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        if self.affine:
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # محاسبه‌ی ضریب به‌روزرسانی آمارهای میانگین/واریانس
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        # اعمال ماسک صرفاً وقتی affine=True باشد
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps
        )


class BatchNorm2d(nn.BatchNorm2d):
    """
    BatchNorm2d با ماسک روی پارامترهای affine (در صورت فعال بودن).
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        if self.affine:
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps
        )


class Identity1d(nn.Module):
    """
    لایه‌ی هویتی 1بعدی با پارامتر قابل یادگیری (و ماسک) برای هر feature.
    - خروجی: input * weight_mask * weight
    """
    def __init__(self, num_features):
        super(Identity1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


class Identity2d(nn.Module):
    """
    لایه‌ی هویتی 2بعدی با پارامتر قابل یادگیری (و ماسک) به شکل (C,1,1).
    - برای اعمال مقیاس کانال‌محور روی feature maps.
    """
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W

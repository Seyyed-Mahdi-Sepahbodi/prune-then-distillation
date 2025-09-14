# پیاده‌سازی هرس SynFlow (بر اساس: https://github.com/ganguli-lab/Synaptic-Flow)
import math
import torch
import numpy as np
import torch.nn as nn

from datasets.data_utils import get_dataloader
from tqdm import tqdm


class Pruner:
    """کلاس پایه‌ی هرس: نگه‌داری ماسک‌ها/امتیازها و اعمال آستانه‌گذاری سراسری یا محلی."""
    def __init__(self, masked_parameters):
        # لیست زوج‌های (mask_tensor, param_tensor) برای پارامترهای قابل‌هرس
        self.masked_parameters = list(masked_parameters)
        # نگه‌داری امتیاز هر پارامتر با کلید id(param) → tensor
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        """محاسبه‌ی امتیاز (باید در زیرکلاس‌ها پیاده‌سازی شود)."""
        raise NotImplementedError

    def _global_mask(self, sparsity):
        """اعمال ماسک با آستانه‌گذاری سراسری روی همه‌ی پارامترها (global unstructured)."""
        # تلفیق همه‌ی امتیازها و تعیین آستانه‌ی مشترک
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        """اعمال ماسک با آستانه‌گذاری محلی (برای هر پارامتر به‌صورت جداگانه)."""
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        """به‌روزرسانی ماسک‌ها بر اساس میزان sparsity و دامنه (scope='global' یا 'local')."""
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        """اعمال ماسک‌ها روی وزن‌ها (صفرکردن وزن‌های حذف‌شده)."""
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        """تنظیم همه‌ی ماسک‌ها روی مقدار ثابتی از alpha (برای تست/دیباگ)."""
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    def stats(self):
        """بازگرداندن تعداد باقیمانده/کل پارامترهای قابل‌هرس (برای گزارش)."""
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    """امتیازدهی تصادفی (پایه‌ی مقایسه)."""
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    """امتیازدهی بر اساس قدرمطلق وزن‌ها (magnitude pruning)."""
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


class SynFlow(Pruner):
    """امتیازدهی SynFlow: |∂L/∂w ⊙ w| روی ورودی همه-یک و شبکه‌ی خطی‌شده."""
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        @torch.no_grad()
        def linearize(model):
            """علامت وزن‌ها را ذخیره و وزن‌ها را قدرمطلق می‌کند (شبکه را «خطی» می‌کند)."""
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            """بازگردانی علامت‌های وزن‌ها به حالت اصلی."""
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        # یک batch «ساختگی» (ورودی همه‌یک) برای عبور رو به جلو و محاسبه‌ی گرادیان
        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(device)

        output = model(input)
        torch.sum(output).backward()

        # محاسبه‌ی امتیاز SynFlow و صفر کردن گرادیان
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)


# ---------- توابع کمکی هرس ----------

def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    """
    بازگرداندن iterator از زوج‌های (mask, param) برای پارامترهای «قابل‌هرس».
    - فقط لایه‌های Conv2d و Linear به‌صورت پیش‌فرض.
    - اگر bias=True، بایاس را هم شامل می‌شود.
    - اگر batchnorm/residual=True، BN و لایه‌های هویتی سفارشی را هم شامل می‌کند.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param


def prunable(module, batchnorm, residual):
    """تشخیص این‌که یک ماژول قابل‌هرس هست یا خیر."""
    isprunable = isinstance(module, (nn.Linear, nn.Conv2d))
    if batchnorm:
        isprunable |= isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    if residual:
        # توجه: این دو کلاس در پروژه به‌صورت سفارشی تعریف شده‌اند (نه در torch.nn)
        isprunable |= isinstance(module, (nn.Identity1d, nn.Identity2d))
    return isprunable


def masks(module):
    """برگرداندن iterator روی همه‌ی بافرهایی که نامشان شامل 'mask' است."""
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf


def prune_loop(model, loss, pruner, dataloader, device, sparsity, prune_epochs, scope, schedule,
               reinitialize=False, train_mode=False):
    """
    حلقه‌ی هرسِ تکرارشونده تا رسیدن به sparsity نهایی:
      - در هر epoch امتیاز محاسبه و ماسک بر اساس برنامه‌ی (schedule) اعمال می‌شود.
      - scope: 'global' یا 'local'
      - schedule: 'exponential' یا 'linear' (در کد از همین دو مقدار استفاده می‌شود)
    """
    # تنظیم حالت مدل (train/eval)
    model.train()
    if not train_mode:
        model.eval()

    # اعمال هرس در چند تکرار با افزایش sparsity
    for epoch in tqdm(range(prune_epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity ** ((epoch + 1) / prune_epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity) * ((epoch + 1) / prune_epochs)
        pruner.mask(sparse, scope)

    # در صورت نیاز، وزن‌ها مجدداً مقداردهی شوند
    if reinitialize:
        model._initialize_weights()

    # بررسی سطح sparsity (اختلاف از هدف)
    remaining_params, total_params = pruner.stats()
    sfscore = np.abs(remaining_params - total_params * sparsity)
    if sfscore >= 5:
        print("Warning: {} prunable parameters remaining, expected {}".format(remaining_params, total_params * sparsity))
        print(f"Score: {sfscore}")


def run(platform, model):
    """
    اجرای هرس SynFlow:
      1) ساخت pruner
      2) تنظیم ابرپارامترها
      3) اجرای حلقه‌ی هرس
    """
    # 1) pruner
    pruner = SynFlow(masked_parameters(model))

    # 2) تنظیمات دیتاست (فقط برای تعیین طول زیرمجموعه‌ی synflow)
    if platform.opt.dataset == 'cifar10':
        num_classes = 10
    elif platform.opt.dataset == 'cifar100':
        num_classes = 100
    elif platform.opt.dataset == 'tiny_imagenet':
        num_classes = 200
    elif platform.opt.dataset == 'imagenet':
        num_classes = 1000
    else:
        raise ValueError("Please check dataset")

    loss = nn.CrossEntropyLoss

    # نسبت sparsity هدف از روی pruning_ratio (همان تبدیل مرسوم SynFlow)
    sparsity = 10 ** (math.log10(1 - platform.opt.pruning_ratio))

    prune_epochs = platform.opt.sf_epochs       # تعداد تکرارهای هرس
    scope = 'global'                            # دامنه‌ی هرس: 'global' یا 'local'
    schedule = 'exponential'                    # برنامه‌ی افزایش sparsity: 'exponential' یا 'linear'
    prune_dataset_ratio = 10                    # طول زیرمجموعه = این نسبت × تعداد کلاس‌ها

    # نکته: امضای get_dataloader → (dataset, batch_size, validation_size=0, dataloader_seed=0, loader_type=None, length=None)
    # در کد اصلی validation_size به‌صورت positional در جای dataloader_seed پاس داده می‌شود (همین رفتار حفظ شده است).
    prune_loader = get_dataloader(
        platform.opt.dataset,
        platform.opt.sf_batch_size,
        platform.opt.validation_size,   # به‌عنوان dataloader_seed عبور می‌کند (مطابق منطق موجود)
        loader_type="synflow",
        length=prune_dataset_ratio * num_classes
    )

    # 3) اجرای هرس
    prune_loop(
        model, loss, pruner, prune_loader, platform.device, sparsity,
        prune_epochs, scope, schedule, reinitialize=False, train_mode=False
    )

import copy
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune

from training import train_runner
from datasets.data_utils import get_dataloader


def iter_pruning_freq_determiner(pruning_ratio):
    """
    تعیین تعداد تکرارهای هرسِ تکرارشونده (Iterative Pruning) بر اساس نسبت ورودی.

    - اگر 0 < pruning_ratio < 1 باشد: تعداد تکرارها طوری انتخاب می‌شود که
      حاصلِ هرس‌های پیاپیِ 20% (هر بار) تا حد ممکن به نسبت هدف نزدیک شود.
      (فرمول تجمعی: 1 - 0.8 ** k)

    - اگر pruning_ratio >= 1 باشد: همان مقدار به‌عنوان تعداد دفعات هرس استفاده می‌شود.

    ورودی:
        pruning_ratio (float): نسبت هدف هرس (یا تعداد دفعات، اگر >= 1)

    خروجی:
        counter (int): تعداد دفعات هرسِ تکرارشونده
    """
    if pruning_ratio < 1:
        counter = 0
        # تکرار تا زمانی که نسبت تجمعی هرس به نسبت هدف برسد/نزدیک شود
        while (1 - np.power(0.8, counter)) <= pruning_ratio:
            counter += 1
            if counter >= 1000:
                print("iter_pruning_freq_determiner error! please check pruning_ratio or this function")
                break

        # انتخاب نزدیک‌ترین k (بالا یا پایین)
        counter_lower = np.abs(pruning_ratio - (1 - np.power(0.8, counter - 1)))
        counter_upper = np.abs(pruning_ratio - (1 - np.power(0.8, counter)))
        return counter - 1 if counter_lower <= counter_upper else counter

    elif pruning_ratio >= 1:
        # در این حالت ورودی به‌عنوان «تعداد دفعات هرس» تفسیر می‌شود
        return pruning_ratio


def get_pruning_module_list(model_for_pruning):
    """
    انتخاب ماژول‌هایی که باید هرس شوند (Conv2d و Linear) و جمع‌آوری آن‌ها
    برای هرسِ «سراسریِ بدون ساختار» (global unstructured).

    خروجی:
        model_for_pruning (nn.Module): مدلِ کپی‌شده (برای اعمال هرسِ PyTorch)
        pruning_module_list (list): لیست زوج‌های (ماژول، 'weight') برای prune.global_unstructured
        pruning_modulename_list (list): لیست نامِ بافرهای ماسکِ متناظر (برای کپی‌کردن ماسک‌ها)
    """
    pruning_module_list = []
    pruning_modulename_list = []

    for name, module in model_for_pruning.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # پیدا کردن بافر ماسک مربوط به وزنِ همان ماژول
            for mask_name, _ in model_for_pruning.named_buffers():
                if f"{name}.weight_mask" in mask_name:
                    pruning_module_list.append((module, 'weight'))
                    pruning_modulename_list.append(mask_name)

    return model_for_pruning, pruning_module_list, pruning_modulename_list


def mask_copy(model, model_for_pruning, pruning_modulename_list):
    """
    کپی‌کردن ماسک‌های هرس‌شده از مدلِ کپی‌شده (که با API هرس شده)
    به مدلِ اصلی (برای حفظ ماسک‌ها در جریان آموزش/ذخیره).

    ورودی:
        model (nn.Module): مدل اصلی
        model_for_pruning (nn.Module): مدل کپی‌شده‌ی هرس‌شده
        pruning_modulename_list (list): نام بافرهای ماسکِ هدف

    خروجی:
        model (nn.Module): مدل اصلی با ماسک‌های به‌روزشده
    """
    for name_copied, mask_copied in model_for_pruning.named_buffers():
        if name_copied in pruning_modulename_list:
            for name_orig, mask_orig in model.named_buffers():
                if name_copied == name_orig:
                    mask_orig.data = mask_copied.data.clone().detach()
    return model


def pruning_loop(platform, model):
    """
    حلقه‌ی «هرس با بازگردانی نرخ‌ِیادگیری» (LR Rewinding):
      1) ساخت یک کپی از مدل و اعمال هرسِ سراسریِ L1 روی وزن‌ها (۲۰٪ در هر تکرار)
      2) کپی‌کردن ماسک‌های به‌دست‌آمده به مدل اصلی
      3) ریزتنظیم (fine-tune) مدل
      4) ذخیره‌ی مدل در هر تکرار

    ورودی:
        platform: شیء پلتفرم با opt، logger و مسیرها
        model (nn.Module): مدل هدف برای هرس
    """
    # تعیین تعداد دفعات هرس بر اساس نسبتِ هدف
    pruning_epoch = iter_pruning_freq_determiner(platform.opt.pruning_ratio)

    # دیتالودرهای پس از هرس (Fine-tuning)
    train_loader, val_loader = get_dataloader(
        platform.opt.dataset,
        platform.opt.post_batch_size,
        platform.opt.validation_size,
        platform.opt.dataloader_seed,
        loader_type="train_val"
    )

    for i in range(pruning_epoch):
        # --- هرس ---
        model_for_pruning = copy.deepcopy(model)
        model_for_pruning, pruning_module_list, pruning_modulename_list = get_pruning_module_list(model_for_pruning)

        # درصد هرس در هر تکرار (۲۰٪ بدون ساختار به‌صورت سراسری)
        sparsity = 0.2

        prune.global_unstructured(
            pruning_module_list,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )

        # کپی ماسک‌های به‌دست‌آمده به مدل اصلی
        model = mask_copy(model, model_for_pruning, pruning_modulename_list)

        # --- ریزتنظیم پس از هرس ---
        train_runner.run(platform, model=model, mode='post', train_loader=train_loader, val_loader=val_loader)

        # --- ذخیره‌ی وزن‌ها پس از هر تکرار ---
        model_temp = copy.deepcopy(model)
        torch.save(
            model_temp.state_dict(),
            os.path.join(
                platform.model_path['result_path'],
                f"post_model_{np.power(0.8, i + 1):.2f}.pth"
            )
        )


def run(platform, model):
    """نقطه‌ی ورودِ ماژول هرس."""
    pruning_loop(platform, model)

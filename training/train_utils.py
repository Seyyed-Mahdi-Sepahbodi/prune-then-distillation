import torch.nn as nn
import torch.optim as optim


def get_lossfunction(type=None):
    """
    انتخاب تابع اتلاف (Loss Function).
    ورودی:
        type (str|None): اگر None باشد CrossEntropyLoss؛ اگر 'MSELoss' باشد MSELoss.
    خروجی:
        criterion (nn.Module): شیء تابع اتلاف PyTorch.
    """
    if type is None:
        # حالت پیش‌فرض: مناسب برای طبقه‌بندی چندکلاسه
        criterion = nn.CrossEntropyLoss()
    elif type == 'MSELoss':
        # حالت MSE: برای رگرسیون یا برخی سناریوهای خاص
        criterion = nn.MSELoss()
    else:
        # نوع نامعتبر
        raise ValueError(f"{type} is not a valid lossfunction type.")
    return criterion


def get_optimizer(opt, model, mode='pre'):
    """
    ساخت بهینه‌ساز بر اساس mode و تنظیمات opt.
    ورودی:
        opt: آرگومان‌های پیکربندی (Namespace) شامل نوع/نرخ یادگیری/Weight Decay.
        model (nn.Module): مدل هدف.
        mode (str): یکی از 'pre'، 'post'، 'kd'.
    خروجی:
        optimizer (torch.optim.Optimizer): شیء بهینه‌ساز PyTorch.
    """
    # انتخاب مجموعه‌تنظیمات بر اساس حالت
    if mode == 'pre':
        optimizer = opt.optimizer
        lr = opt.lr
        weight_decay = opt.weight_decay
    elif mode == 'post':
        optimizer = opt.post_optimizer
        lr = opt.post_lr
        weight_decay = opt.post_weight_decay
    elif mode == 'kd':
        optimizer = opt.kd_optimizer
        lr = opt.kd_lr
        weight_decay = opt.kd_weight_decay
    else:
        raise ValueError(f"Please check the mode option (Please choose in (pre, post, kd) not {mode})")

    # ساخت بهینه‌ساز بر اساس نوع انتخاب‌شده
    if optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'momentum':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer == 'nesterov':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    # توجه: در صورت نوع نامعتبر، تابع چیزی برنمی‌گرداند (همان منطق اصلی حفظ شده است).


def get_scheduler(opt, optimizer, mode='pre'):
    """
    ساخت زمان‌بندِ نرخ یادگیری (MultiStepLR) بر اساس mode و تنظیمات opt.
    ورودی:
        opt: آرگومان‌های پیکربندی شامل نقاط افت و نرخ افت.
        optimizer: بهینه‌ساز ساخته‌شده.
        mode (str): یکی از 'pre'، 'post'، 'kd'.
    خروجی:
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): زمان‌بند LR.
    """
    # استخراج پارامترهای برنامه‌ی افت LR بر اساس حالت
    if mode == 'pre':
        lr_drops = opt.lr_drops
        lr_drop_rate = opt.lr_drop_rate
    elif mode == 'post':
        lr_drops = opt.post_lr_drops
        lr_drop_rate = opt.post_lr_drop_rate
    elif mode == 'kd':
        lr_drops = opt.kd_lr_drops
        lr_drop_rate = opt.kd_lr_drop_rate
    else:
        raise ValueError(f"Please check the mode option (Please choose in (pre, post, kd) not {mode})")

    # MultiStepLR: در epochهای مشخص، LR با ضریب gamma ضرب می‌شود
    return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_drops, gamma=lr_drop_rate)

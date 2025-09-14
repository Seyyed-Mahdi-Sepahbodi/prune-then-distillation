import torch.nn as nn

# مجموعه‌ای از توابع مقداردهی اولیه (Initialization) برای وزن‌ها/بایاس‌ها.
def xavier_normal_weight(model, layer='linear'):
    """
    اعمال Xavier Normal روی وزنِ لایه‌های مشخص‌شده.
    پارامترها:
        model: شیء مدل PyTorch
        layer (str): 'linear' یا 'conv' برای مشخص‌کردن نوع لایه هدف
    """
    for _, module in model.named_modules():
        if layer == 'linear':
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
        elif layer == 'conv':
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
        else:
            # اگر نوع لایه اشتباه باشد، خطا می‌دهد (همین رفتار حفظ شده است)
            raise ValueError(f"Please check your layer type: linear or conv. (Your input is {layer})")


def xavier_normal_bias(model):
    """
    اعمال Xavier Normal روی بایاسِ لایه‌های Linear.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.bias)


def xavier_uniform_weight(model, layer='linear'):
    """
    اعمال Xavier Uniform روی وزنِ لایه‌های مشخص‌شده.
    ⚠️ توجه: ساختار if/else دقیقاً مطابق نسخه‌ی اصلی نگه داشته شده است.
    """
    for _, module in model.named_modules():
        if layer == 'linear':
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        if layer == 'conv':
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
        else:
            # رفتار نسخه‌ی اصلی: در غیر 'conv' بودن، خطا می‌دهد.
            # (حفظ منطق، حتی اگر در عمل محدودکننده باشد)
            raise ValueError(f"Please check your layer type: linear or conv. (Your input is {layer})")


def xavier_uniform_bias(model):
    """
    اعمال Xavier Uniform روی بایاسِ لایه‌های Linear.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.bias)


def zeros_weight(model):
    """
    مقداردهی وزنِ لایه‌های Linear با صفر.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)


def zeros_bias(model):
    """
    مقداردهی بایاسِ لایه‌های Linear با صفر.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias)


def normal_weight(model):
    """
    مقداردهی وزنِ لایه‌های Linear با توزیع نرمال پیش‌فرض PyTorch.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)


def normal_bias(model):
    """
    مقداردهی بایاسِ لایه‌های Linear با توزیع نرمال پیش‌فرض PyTorch.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.bias)

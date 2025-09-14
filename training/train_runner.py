import os
import torch

from models.get_models import get_model
from training import train_model


def run(platform, **kwargs):
    """
    اجرای حلقه‌ی آموزش با توجه به mode:
      - mode='pre'  → ساخت مدل جدید طبق تنظیمات و آموزشِ پیش‌از هر کاری
      - mode='post' → آموزشِ پس از هرس/فرآیندهای دیگر روی مدلِ ورودی

    ورودی‌ها:
        platform: شیء پلتفرم شامل opt (تنظیمات)، مسیرها و logger
        **kwargs:
            - mode: 'pre' یا 'post' (اجباری)
            - model: فقط وقتی mode='post' است باید مدل را بدهید
    """
    if "mode" not in kwargs:
        raise ValueError("Must put mode value (pre or post)")

    if kwargs["mode"] == "pre":
        # ساخت مدل بر اساس نوع/دیتاست و انتقال به device
        model = get_model(platform.opt.model_type, platform.opt.dataset).to(platform.device)
        epochs = platform.opt.epochs
        batch_size = platform.opt.batch_size

    elif kwargs["mode"] == "post":
        # استفاده از مدلِ از قبل ساخته‌شده/هرس‌شده
        model = kwargs["model"]
        epochs = platform.opt.post_epochs
        batch_size = platform.opt.post_batch_size

    # شمارش وزن‌های باقیمانده و لاگ جدول اولیه
    model.weight_counter()
    model.logging_table(platform.logger)

    # اجرای حلقه‌ی آموزش (بر اساس mode و مقادیر epochs/batch_size محاسبه‌شده)
    train_model.run(platform, model, epochs, batch_size, kwargs["mode"])

    # ذخیره‌ی وزن‌های مدلِ آموزش‌دیده در حالت pre-train
    if kwargs["mode"] == 'pre':
        torch.save(model.state_dict(), os.path.join(platform.model_path["result_path"], "pre_model.pth"))

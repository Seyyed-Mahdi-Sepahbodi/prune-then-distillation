import os
import torch

from models.get_models import get_model
from pruning import lr_rewinding, synflow
from training import train_runner


def run(platform):
    """
    اجرای حلقه‌ی هرس بر اساس تنظیمات platform.opt.
    منطق اصلی دست‌نخورده است؛ فقط توضیحات اضافه شده‌اند.
    """
    # اعلان شروع هرس با نوع انتخاب‌شده (lr_rewinding یا synflow)
    platform.logger.logger.info(f'Start pruning with {platform.opt.pruner}...')

    if platform.opt.pruner == "lr_rewinding":
        # 1) ساخت مدل و بارگذاری وزن‌های آموزش‌دیده (pretrained)
        model = get_model(platform.opt.model_type, platform.opt.dataset).to(platform.device)
        model.load_state_dict(torch.load(platform.model_path["trained_model_path"]))

        # 2) شمارش وزن‌های باقیمانده/کل و لاگ جدول اولیه
        model.weight_counter()
        model.logging_table(platform.logger)

        # 3) اجرای هرس به روش LR Rewinding
        lr_rewinding.run(platform, model)

    elif platform.opt.pruner == "synflow":
        # 1) ساخت مدل (بدون بارگذاری وزن‌های قبلی؛ مطابق منطق موجود)
        model = get_model(platform.opt.model_type, platform.opt.dataset).to(platform.device)

        # 2) شمارش وزن‌ها و لاگ جدول اولیه
        model.weight_counter()
        model.logging_table(platform.logger)

        # 3) اجرای هرس به روش SynFlow
        synflow.run(platform, model)

        # 4) آموزش پساهرس (Fine-tuning پس از SynFlow)
        train_runner.run(platform, model=model, mode='post')

    # شمارش/گزارش نهایی وضعیت وزن‌ها پس از هرس
    model.weight_counter()
    platform.logger.logger.info("Pruning Done!\n\n")

    # ذخیره‌ی مدل نهایی پس از هرس
    torch.save(model.state_dict(), os.path.join(platform.model_path['result_path'], "post_model.pth"))

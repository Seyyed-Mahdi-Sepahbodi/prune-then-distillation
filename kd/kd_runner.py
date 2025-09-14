import torch
from kd import kd_vanilla
from models.get_models import get_model


def run(platform):
    """
    اجرای دانش-تقطیر (Knowledge Distillation) بر اساس تنظیمات موجود در platform.opt
    
    ورودی:
        platform (object): شامل تنظیمات (opt)، مسیرها (model_path)، دستگاه (device)، و logger.
    """

    # شروع فرآیند KD (نوع KD در opt مشخص شده: مثل 'vanilla')
    platform.logger.logger.info(
        f'Start knowledge distillation ({platform.opt.kd_type})...'
    )

    # بارگذاری مدل معلم (teacher)
    teacher_model = get_model(
        platform.opt.teacher_model_type,
        platform.opt.dataset
    ).to(platform.device)

    teacher_model.load_state_dict(
        torch.load(platform.model_path["teacher_model_path"],
                   map_location=platform.device)
    )

    # ساخت مدل دانش‌آموز (student)
    if "custom" in platform.opt.student_model_type:
        # اگر نوع دانش‌آموز custom باشد → از طرح (plan) معلم برای ساختش استفاده می‌کند
        model = get_model(
            platform.opt.student_model_type,
            platform.opt.dataset,
            plan=teacher_model.get_student_plan()
        ).to(platform.device)
    else:
        # در حالت عادی → مدل دانش‌آموز بر اساس نوع/دیتاست ساخته می‌شود
        model = get_model(
            platform.opt.student_model_type,
            platform.opt.dataset
        ).to(platform.device)

    # شمارش وزن‌ها و ثبت جدول ساختار مدل برای گزارش‌گیری
    model.weight_counter()
    model.logging_table(platform.logger)

    # اجرای KD با روش vanilla
    if platform.opt.kd_type == 'vanilla':
        kd_vanilla.run(platform, model, teacher_model)

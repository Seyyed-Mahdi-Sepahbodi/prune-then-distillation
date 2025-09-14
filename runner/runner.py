from pruning import prune_runner
from training import train_runner
from kd import kd_runner


def run(platform):
    """
    تابع اصلی اجرای فرایندها.
    ورودی:
        platform: شیء پلتفرم شامل تنظیمات (opt) و logger.
    جریان کلی:
      1) پیش‌آموزش (اختیاری)
      2) هرس (اختیاری)
      3) دانش‌تقطیر (اختیاری)
    """

    # --- 1) پیش‌آموزش مدل (pre-train) در صورت فعال بودن ---
    if platform.opt.pre_train is True:
        # اجرای حلقه‌ی آموزش پیشین (مثلاً آموزش از ابتدا یا warmup)
        train_runner.run(platform, mode='pre')

    # --- 2) هرس (pruning) در صورت فعال بودن ---
    if platform.opt.pruning is True:
        # اجرای روال هرس طبق pruner انتخاب‌شده در تنظیمات
        prune_runner.run(platform)

    # --- 3) دانش‌تقطیر (knowledge distillation) در صورت فعال بودن ---
    if platform.opt.kd is True:
        # اجرای حلقه‌ی KD (مثلاً vanilla KD) با تنظیمات موجود
        kd_runner.run(platform)

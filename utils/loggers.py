import logging
import os
from prettytable import PrettyTable


class Logger:
    """
    رَپر ساده روی logging برای:
      - نوشتن خروجی روی کنسول و فایل (یا فقط فایل در حالت test)
      - چاپ جدول تنظیمات (args) با PrettyTable در ابتدای اجرا
    """
    def __init__(self, opt, result_path, mode=None, logger_type=None):
        self.result_path = result_path        # مسیر ذخیره‌ی لاگ‌ها
        self.mode = mode                      # حالت اجرا (None یا 'test')
        self.logger_type = logger_type        # نامِ logger سفارشی (در غیر این صورت از روت استفاده می‌شود)

        if mode is None:
            # حالت عادی: لاگ به کنسول + فایل و چاپ جدول پارامترها
            self.logger = self.init_logging()
            self.parser_logging(opt, self.logger)
        elif mode == 'test':
            # حالت تست: فقط لاگ به فایل (بدون استریم به کنسول)
            self.logger = self.init_logging(mode='test')
            # جلوگیری از انتشار لاگ به loggers والد (برای تست‌های ایزوله)
            self.logger.propagate = False

    def init_logging(self, mode=None):
        """
        مقداردهی logging:
          - اگر logger_type تعیین شده باشد، همان logger نام‌دار استفاده می‌شود
          - سطح لاگ: INFO
          - اگر mode != 'test' → استریم به کنسول نیز اضافه می‌شود
          - همیشه یک FileHandler روی result_path/test_log.log اضافه می‌شود
        """
        # انتخاب logger (نام‌دار یا روت)
        logger = logging.getLogger(self.logger_type) if self.logger_type is not None else logging.getLogger()

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')

        # افزودن استریم‌هندر برای چاپ روی کنسول (به‌جز در حالت test)
        if mode != "test":
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        # هندلر فایل: لاگ‌ها در فایل test_log.log نوشته می‌شوند (overwrite هر بار)
        file_handler = logging.FileHandler(os.path.join(self.result_path, "test_log.log"), mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def close_logging_handler(self):
        """
        بستن و حذف همه‌ی هندلرهای logger (برای اجرای تست‌های جدید بدون تداخل هندلرهای قبلی).
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def parser_logging(self, opt, logger):
        """
        چاپ همه‌ی آرگومان‌های parser به‌صورت جدول در لاگ.
        """
        table = PrettyTable(["Item", "Value"])
        for arg in vars(opt):
            table.add_row([arg, getattr(opt, arg)])
        logger.info(table)

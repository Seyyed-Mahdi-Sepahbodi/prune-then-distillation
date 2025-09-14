import torch.nn as nn
from prettytable import PrettyTable


class base_model(nn.Module):
    """
    کلاس پایه‌ی مدل‌ها با قابلیت:
      - شمارش وزن‌های باقیمانده/هرس‌شده (وقتی ماسکِ weight_mask روی لایه‌ها وجود دارد)
      - ساخت جدول خلاصه با PrettyTable
      - نگه‌داری آرایه‌های ثبتِ لاگِ آموزش/اعتبارسنجی/تست
      - استخراج «طرح» (plan) برای دانش‌آموز از روی معلم (برای خانواده‌ی VGG)
    """

    def __init__(self):
        super(base_model, self).__init__()
        # نوع مدل (مثلاً 'vgg19') – باید در کلاس فرزند مقداردهی شود
        self.model_type = None
        # نگه‌دارنده‌ی طرح (در صورت استفاده)
        self.plan = None

        # شمارنده‌های وزن‌های باقیمانده/کل (ابتدائاً 1 تا نسبت NaN نشود)
        self.remaining_params, self.total_params = 1, 1
        self.weight_ratio = self.remaining_params / self.total_params * 100

        # جدول PrettyTable برای گزارش
        self.table = None

        # آرایه‌های ثبت روند آموزش/تست/اعتبارسنجی
        self.train_trainset_loss_arr = [0]
        self.train_testset_loss_arr = [0]
        self.train_testset_accuracy_arr = [0]
        self.val_loss_arr = []
        self.val_accuracy_arr = []

    def weight_counter(self):
        """
        تمام زیرماژول‌ها را پیمایش می‌کند، اگر Linear یا Conv2d بودند و باندلِ 'weight_mask' داشتند،
        تعداد المان‌های 1 ماسک را به‌عنوان «باقیمانده» می‌شمارد و در جدول ذخیره می‌کند.
        در پایان یک ردیفِ *total* هم اضافه می‌شود.
        """
        self.table_initialize()
        self.remaining_params, self.total_params = 0, 0

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) is True:
                self.check_mask_weight(name, module)

        # نسبت درصدیِ وزن‌های باقیمانده
        self.weight_ratio = self.remaining_params / self.total_params * 100
        self.table.add_row([
            '*total*',
            f"{self.total_params:.0f}",
            f"{self.remaining_params:.0f} / {self.total_params - self.remaining_params:.0f}",
            f"{self.weight_ratio:.2f}"
        ])

    def check_mask_weight(self, name, module):
        """
        در بافرهای ماژول به دنبال 'weight_mask' می‌گردد و:
          - مجموع 1ها را (weightهای حفظ‌شده) به‌عنوان remaining می‌شمارد
          - تعداد کل المان‌ها را به‌عنوان total می‌گیرد
          - هر لایه را به جدول اضافه می‌کند
        """
        for buf_name, buf_param in module.named_buffers():
            if "weight_mask" in buf_name:
                remaing_p = buf_param.detach().cpu().numpy().sum()
                total_p = buf_param.numel()
                self.remaining_params += remaing_p
                self.total_params += total_p
                self.table.add_row([
                    name,
                    f"{total_p:.0f}",
                    f"{remaing_p:.0f} / {total_p - remaing_p:.0f}",
                    f"{remaing_p / total_p * 100:.2f}"
                ])

    def get_teacher_plan_base(self):
        """
        پایه‌ی طرح: برای هر لایه‌ی Linear/Conv2d، زوج (remaining_params, total_params) را
        با توجه به weight_mask استخراج می‌کند و به صورت لیستی بازمی‌گرداند.
        """
        plan_base = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) is True:
                remaining_params, total_params = self.mask_weight_for_st(module)
                plan_base.append((remaining_params, total_params))
        return plan_base

    def mask_weight_for_st(self, module):
        """
        از روی weight_mask های ماژول، مجموع remaining و total را برمی‌گرداند.
        """
        remaining_params, total_params = 0, 0
        for buf_name, buf_param in module.named_buffers():
            if "weight_mask" in buf_name:
                remaining_params += buf_param.detach().cpu().numpy().sum()
                total_params += buf_param.numel()
        return remaining_params, total_params

    def get_student_plan(self, plan_base):
        """
        تولید «طرح» کانال‌ها برای دانش‌آموز (فقط برای معماری‌های VGG).
        مقدارهای 0 بعداً با تعداد کانال‌ها جایگزین می‌شوند و 'M' نشان‌دهنده‌ی MaxPool است.
        """
        plans = {
            'vgg11': [0, 'M', 0, 'M', 0, 0, 'M', 0, 0, 'M', 0, 0],
            'vgg13': [0, 0, 'M', 0, 0, 'M', 0, 0, 'M', 0, 0, 'M', 0, 0],
            'vgg16': [0, 0, 'M', 0, 0, 'M', 0, 0, 0, 'M', 0, 0, 0, 'M', 0, 0, 0],
            'vgg19': [0, 0, 'M', 0, 0, 'M', 0, 0, 0, 0, 'M', 0, 0, 0, 0, 'M', 0, 0, 0, 0]
        }

        plan = plans[self.model_type]

        if "vgg" in self.model_type:
            # مقداردهی کانال‌های لایه‌ی اول از روی نسبت باقی‌مانده‌ها
            plan_init = plan_base.pop(0)
            plan[0] = round(plan_init[0] / (9 * 3))  # 3x3 kernel, 3 in-ch for RGB
            plan_temp = plan[0]

            # پرکردن سایر 0ها بر اساس نسبت باقیمانده‌ی هر لایه
            for layer_info in enumerate(plan):
                if layer_info[1] == 0:
                    plan_data = plan_base.pop(0)
                    plan[layer_info[0]] = round(plan_data[0] / (plan_temp * 3))
                    plan_temp = plan[layer_info[0]]
        else:
            # برای معماری‌های غیر VGG، طرحِ پیش‌فرض
            plan = [0]

        return plan

    def get_student_plan(self):
        """
        نسخه‌ی بدون ورودی: ابتدا plan_base را از معلم استخراج می‌کند،
        سپس طرح نهایی دانش‌آموز را برمی‌گرداند.
        """
        plan_base = self.get_teacher_plan_base()
        plan = self.get_student_plan(plan_base)
        return plan

    def table_initialize(self):
        """
        آماده‌سازی جدول PrettyTable برای گزارشِ شمارش وزن‌ها.
        """
        self.table = PrettyTable(['Layer', 'Total_Weight', 'Remaining/Pruned', 'Ratio(%)'])
        self.table.align["Layer"] = "l"
        self.table.align["Total_Weight"] = "l"
        self.table.align["Remaining/Pruned"] = "l"
        self.table.align["Ratio(%)"] = "l"

    def logging_table(self, logger):
        """
        لاگ‌کردن جدول ساخته‌شده توسط logger بیرونی.
        """
        logger.logger.info(self.table)

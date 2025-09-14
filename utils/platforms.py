import argparse
import json
import os
import torch
import numpy as np
from datetime import datetime

from utils.loggers import Logger
from utils.base_utils import set_cuda, make_dir


class Platform:
    """
    سکو/پلتفرم اجرای پروژه:
      - پارس‌کردن آرگومان‌ها و بارگذاری هایپرپارامترها
      - آماده‌سازی مسیرهای ذخیره‌سازی نتایج/مدل‌ها
      - تنظیم GPU/CPU
      - راه‌اندازی logger و ذخیره‌ی هایپرپارامترها
    """
    def __init__(self, mode=None, logger_type=None):
        self.opt = self.get_parser()                 # آرگومان‌ها از CLI
        self.start_setup(mode)                       # تنظیمات اولیه (seed و نام مسیر نتایج)
        self.model_path = self.get_model_path()      # مسیرهای مدل/نتیجه
        make_dir(self.model_path["result_path"])     # ساخت پوشه‌ی نتایج
        self.device = set_cuda(self.opt.GPU_num)     # تنظیم دیوایس
        self.logger = Logger(self.opt, self.model_path["result_path"], mode, logger_type)  # راه‌اندازی لاگر

        # ذخیره‌ی هایپرپارامترها به‌صورت JSON در پوشه‌ی نتایج
        with open(f"{self.model_path['result_path']}/hyperparameter.json", "w") as json_file:
            json.dump(vars(self.opt), json_file, indent=4)

    def get_parser(self):
        """
        ساخت و مقداردهی آرگومان‌های خط فرمان.
        """
        parser = argparse.ArgumentParser(
            description='KD-Pruning',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # --- Base ---
        base_args = parser.add_argument_group('Base hyperparameters')
        base_args.add_argument("--experiment_name", default=None, type=str, help="نام آزمایش/خروجی")
        base_args.add_argument("--seed", default=None, type=int, help="بذر تصادفی (None یعنی عدم اعمال)")
        base_args.add_argument("--dataloader_seed", default=None, type=int, help="بذر تصادفی دیتالودر")
        base_args.add_argument("--GPU_num", default=None, type=int, help="شماره GPU")
        base_args.add_argument("--hparam_fname", default=None, type=str, nargs='*',
                               help="نام فایل(های) هایپرپارامتر در experiments/hyperparam بدون پسوند")
        base_args.add_argument("--model_type", default=None, type=str, help="نوع مدل")
        base_args.add_argument("--dataset", default=None, type=str, choices=('cifar10', 'cifar100', 'imagenet'),
                               help="نام دیتاست")
        base_args.add_argument("--batch_size", default=None, type=int, help="اندازه‌ی مینی‌بتچ")
        base_args.add_argument("--validation_size", default=None, type=int, help="اندازه‌ی مجموعه‌ی اعتبارسنجی")

        # --- Pre-training ---
        pre_args = parser.add_argument_group('Pre training hyperparameters')
        pre_args.add_argument("--pre_train", action='store_true', help="اجرای فاز پیش‌آموزش")
        pre_args.add_argument("--epochs", default=None, type=int)
        pre_args.add_argument("--lr", default=None, type=float, help="نرخ یادگیری")
        pre_args.add_argument("--lr_drops", default=None, type=int, nargs='*', help="اپک‌های کاهش LR")
        pre_args.add_argument("--lr_drop_rate", default=None, type=float, help="ضریب کاهش LR")
        pre_args.add_argument("--optimizer", default=None, type=str, choices=('adam', 'sgd', 'momentum'),
                              help="نوع بهینه‌ساز")
        pre_args.add_argument("--weight_decay", default=None, type=float, help="وزن پنالتی (WD)")

        # --- Pruning ---
        prune_args = parser.add_argument_group('Pruning hyperparameters')
        prune_args.add_argument("--pruning", action='store_true', help="اجرای فاز هرس")
        prune_args.add_argument("--pre_model_name", default=None, type=str, help="نام مدل پیش‌آموزش‌داده‌شده")
        prune_args.add_argument("--pruner", default=None, type=str, choices=('l1norm', 'random', 'lr_rewinding'),
                                help="نوع هرسی که اجرا می‌شود")
        prune_args.add_argument("--pruning_ratio", default=None, type=float, help="نسبت حذف وزن‌ها")
        prune_args.add_argument("--post_batch_size", default=None, type=int, help="اندازه‌ی مینی‌بتچِ پساآموزش")
        prune_args.add_argument("--post_epochs", default=None, type=int)
        prune_args.add_argument("--post_lr", default=None, type=float, help="نرخ یادگیریِ پساآموزش")
        prune_args.add_argument("--post_lr_drops", default=None, type=int, nargs='*', help="اپک‌های کاهش LR پساآموزش")
        prune_args.add_argument("--post_lr_drop_rate", default=None, type=float, help="ضریب کاهش LR پساآموزش")
        prune_args.add_argument("--post_optimizer", default=None, type=str,
                                choices=('adam', 'sgd', 'momentum', 'neterov'),
                                help="نوع بهینه‌ساز در پساآموزش")
        prune_args.add_argument("--post_weight_decay", default=None, type=float, help="WD پساآموزش")

        # --- Knowledge Distillation ---
        kd_args = parser.add_argument_group('KD hyperparameters')
        kd_args.add_argument("--kd", action='store_true', help="اجرای فاز دانش‌تقطیر")
        kd_args.add_argument("--kd_type", default=None, type=str, choices=('vanilla',), help="نوع KD")
        kd_args.add_argument("--teacher_model_name", default=None, type=str,
                             help="نام فایل مدل معلم در experiments/teacher_model (بدون پسوند)")
        kd_args.add_argument("--student_model_type", default=None, type=str, help="نوع مدل دانش‌آموز")
        kd_args.add_argument("--kd_epochs", default=None, type=int)
        kd_args.add_argument("--kd_batch_size", default=None, type=int, help="اندازه‌ی مینی‌بتچ KD")
        kd_args.add_argument("--kd_lr", default=None, type=float, help="نرخ یادگیری KD")
        kd_args.add_argument("--kd_lr_drops", default=None, type=int, nargs='*', help="اپک‌های کاهش LR در KD")
        kd_args.add_argument("--kd_lr_drop_rate", default=None, type=float, help="ضریب کاهش LR در KD")
        kd_args.add_argument("--kd_optimizer", default=None, type=str, choices=('adam', 'sgd', 'momentum'),
                             help="نوع بهینه‌ساز KD")
        kd_args.add_argument("--kd_weight_decay", default=None, type=float, help="WD در KD")
        kd_args.add_argument("--kd_alpha", default=None, type=float, help="ضریب ترکیب KD")
        kd_args.add_argument("--kd_temp", default=None, type=float, help="دما در KD")

        opt = parser.parse_args()
        return opt

    def start_setup(self, mode):
        """
        راه‌اندازی اولیه بر اساس حالت اجرا:
          - حالت عادی: مقداردهی هایپرپارامترها از فایل‌ها + تنظیم seed
          - حالت 'test': تنظیم نام خروجی unittest
        """
        if mode is None:
            self.hparam_setup()
            self.set_seed()
        elif mode == 'test':
            self.opt.result_fname = "unittest"
        else:
            raise ValueError("Please check mode option value")

        # اگر نام آزمایش مشخص نشده باشد، نام پوشه‌ی نتایج را بر اساس تاریخ/زمان می‌سازد
        if self.opt.experiment_name is None:
            self.opt.experiment_name = self.get_result_path_name()

    def get_result_path_name(self):
        """
        تولید نام مسیر ذخیره‌ی نتایج به فرم YYMMDD_hhmmss.
        """
        save_path = datetime.today().strftime("%y%m%d_%H%M%S")
        return save_path

    def get_model_path(self):
        """
        ساخت دیکشنری مسیرهای ذخیره/بارگذاری مدل‌ها بسته به تنظیمات فعلی.
        """
        model_path = {}
        model_path["result_path"] = os.path.join("result/data", self.opt.experiment_name)

        # مسیر ذخیره‌سازی مدلِ پیش‌آموزش
        if self.opt.pre_train is True:
            model_path["pre_train_model_path"] = model_path["result_path"]

        # تنظیم مسیرهای مربوط به هرس
        if self.opt.pruning is True:
            # مسیر بارگذاری مدلِ آموزش‌داده‌شده برای شروع هرس
            if self.opt.pre_train is True:
                model_path["trained_model_path"] = os.path.join(model_path["result_path"], "pre_model.pth")
            else:
                model_path["trained_model_path"] = f"experiments/trained_model/{self.opt.pre_model_name}.pth"
                self.model_check(model_path["trained_model_path"])
            # مسیر ذخیره‌ی مدلِ پس از هرس
            model_path["pruned_model_path"] = os.path.join(model_path["result_path"], "post_model.pth")

        # مسیرهای مربوط به KD (مدل معلم)
        if self.opt.kd is True:
            if self.opt.pruning is True:
                # اگر هرس انجام شده، مدل معلم همان مدل هرسی است
                model_path["teacher_model_path"] = model_path["pruned_model_path"]
            else:
                # در غیر این صورت از مسیر سفارشیِ تعیین‌شده استفاده می‌شود (منطق موجود حفظ شده است)
                model_path["teacher_model_path"] = (
                    f"D:/College/EmbeddedAI/prune-then-distill/experiments/teacher_model/"
                    f"{self.opt.teacher_model_name}.pth"
                )
                self.model_check(model_path["teacher_model_path"])

        return model_path

    def model_check(self, PATH):
        """
        بررسی وجود فایل مدل در مسیر داده‌شده؛ در غیر این صورت خطا می‌دهد.
        """
        if os.path.isfile(PATH) is False:
            raise ValueError(f"Pre trained model is not exist! (Put in experiments/pre_model/{PATH})")

    def set_seed(self):
        """
        تنظیم بذرهای تصادفی برای تکرارپذیری (در صورت تعیین opt.seed).
        """
        if self.opt.seed is not None:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed_all(self.opt.seed)
            np.random.seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def hparam_setup(self):
        """
        مقداردهی خودکار opt از روی فایل‌های hyperparam بر اساس ماژول‌های فعال (train/prune/kd).
        اگر hparam_fname مشخص نشده باشد، از مقادیر پیش‌فرض همان ماژول‌ها استفاده می‌شود.
        """
        param_folder_path = "experiments/hyperparam"
        param_fpath = []

        # انتخاب فایل(ها)ی هایپرپارامتر
        if self.opt.hparam_fname is None:
            if self.opt.pre_train is True:
                param_fpath.append("train_default")
            if self.opt.pruning is True:
                param_fpath.append("prune_default")
            if self.opt.kd is True:
                param_fpath.append("kd_default")
        else:
            param_fpath = self.opt.hparam_fname

        # تبدیل به مسیر کامل
        for index, path in enumerate(param_fpath):
            param_fpath[index] = os.path.join(param_folder_path, f"{path}.json")

        # بررسی وجود فایل‌ها
        for path in param_fpath:
            if os.path.exists(path) is False:
                raise ValueError(f"{self.opt.hparam_fname}.json is not exists!")

        # بارگذاری و ادغام مقادیر در opt (فقط کلیدهایی که هنوز None هستند)
        for path in param_fpath:
            with open(path, "rt") as f:
                hparam = argparse.Namespace()
                hparam.__dict__.update(json.load(f))
            for key in vars(hparam).keys():
                if self.opt.__dict__[key] is None:
                    self.opt.__dict__[key] = hparam.__dict__[key]

        # تعیین نوع مدل معلم
        if self.opt.kd is True:
            if self.opt.pruning is False:
                # وقتی مدل معلم از فایل خارجی بار می‌شود، نوع مدل از نام فایل استخراج می‌گردد
                teacher_model_name = self.opt.teacher_model_name.split('_')
                self.opt.teacher_model_type = teacher_model_name[0]
            else:
                # وقتی KD پس از هرس خودمان است، نوع معلم همان نوع مدل فعلی است
                self.opt.teacher_model_type = self.opt.model_type

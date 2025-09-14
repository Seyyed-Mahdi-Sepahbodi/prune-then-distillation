import json
import os
import torch


def set_cuda(GPU_num):
    """
    تنظیم وضعیت CUDA و برگرداندن شیء device.
    ورودی:
        GPU_num (str): شماره‌ی GPU مورد استفاده (0، 1، 2، ...).
    خروجی:
        device (torch.device): دیوایس انتخاب‌شده (cuda یا cpu).
    """
    if torch.cuda.is_available():
        # اگر CUDA در دسترس باشد، GPU مشخص‌شده را فعال می‌کنیم
        device = torch.device(f'cuda:{GPU_num}')
        torch.cuda.set_device(device)
    else:
        # در غیر این صورت از CPU استفاده می‌شود
        device = torch.device("cpu")
    return device


def make_dir(path):
    """
    ساخت پوشه برای ذخیره‌ی نتایج آزمایش.
    اگر پوشه از قبل وجود داشته باشد، خطا می‌دهد تا از overwrite جلوگیری شود.
    ورودی:
        path (str): مسیر ذخیره‌سازی نتایج.
    """
    if os.path.exists(path) is False:
        os.makedirs(path)
    else:
        raise ValueError("Error: folder already exists. Delete folder or set other --experiment_name")


def save_result_data(model, path, mode="pre"):
    """
    ذخیره‌ی آرایه‌های دقت/اتلاف ذخیره‌شده در مدل به‌صورت فایل‌های JSON در مسیر نتایج.
    ورودی:
        model: شیء مدل که آرایه‌های نتایج در آن نگه‌داری می‌شوند.
        path (str): مسیر پوشه‌ی نتایج.
        mode (str): پیشوند فایل‌ها برای تفکیک حالت‌ها (pre/post/kd و ...).
    """
    with open(f"{path}/{mode}_train_trainset_loss.json", "w") as json_file:
        json.dump(model.train_trainset_loss_arr, json_file, indent=4)
    with open(f"{path}/{mode}_train_testset_accu.json", "w") as json_file:
        json.dump(model.train_testset_accuracy_arr, json_file, indent=4)
    with open(f"{path}/{mode}_train_testset_loss.json", "w") as json_file:
        json.dump(model.train_testset_loss_arr, json_file, indent=4)
    with open(f"{path}/{mode}_val_loss.json", "w") as json_file:
        json.dump(model.val_loss_arr, json_file, indent=4)
    with open(f"{path}/{mode}_val_accu.json", "w") as json_file:
        json.dump(model.val_accuracy_arr, json_file, indent=4)


def result_data_initializer(model):
    """
    ریست‌کردن آرایه‌های دقت/اتلاف در مدل پس از ذخیره، برای اجرای آزمایش‌های بعدی.
    ورودی:
        model: شیء مدل که آرایه‌های نتایج در آن نگه‌داری می‌شوند.
    """
    model.train_trainset_loss_arr = [0]
    model.train_testset_loss_arr = [0]
    model.train_testset_accuracy_arr = [0]
    model.val_loss_arr = []
    model.val_accuracy_arr = []

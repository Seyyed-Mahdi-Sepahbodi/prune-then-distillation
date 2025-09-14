import os
import torch

from datasets.data_utils import get_dataloader
from tqdm import tqdm
from training.train_utils import get_optimizer, get_scheduler, get_lossfunction
from training.Tester import Tester
from utils.base_utils import save_result_data, result_data_initializer


def train(model, device, train_loader, optimizer, criterion):
    """
    حلقه‌ی آموزش عمومی PyTorch برای یک epoch.

    ورودی‌ها:
        model (torch.nn.Module): مدل مورد آموزش.
        device (torch.device): دیوایس اجرا (cpu/cuda).
        train_loader (DataLoader): داده‌های آموزش.
        optimizer (torch.optim.Optimizer): بهینه‌ساز.
        criterion (nn.Module): تابع اتلاف.

    خروجی:
        training_loss (float): میانگین اتلاف آموزشی (بر حسب نمونه).
    """
    model.train()
    training_loss = 0.0

    for batch_idx, (data, label) in enumerate(train_loader):
        # انتقال داده به دیوایس
        data, label = data.to(device), label.to(device)

        # صفر کردن گرادیان‌ها
        optimizer.zero_grad()

        # پیشروی، محاسبه‌ی اتلاف، انباشتن مقدار
        outputs = model(data)
        loss = criterion(outputs, label)
        training_loss += loss.item()

        # پس‌انتشار و به‌روزرسانی وزن‌ها
        loss.backward()
        optimizer.step()

    # تبدیل مجموع اتلاف مینی‌بتچ‌ها به میانگین بر حسب تعداد نمونه
    training_loss /= len(train_loader.dataset)

    # ثبت اتلاف برای گزارش‌گیری
    model.train_trainset_loss_arr.append(training_loss)
    return training_loss


def run(platform, model, epochs, batch_size, mode):
    """
    حلقه‌ی کامل آموزش (epochs بار) با ارزیابی و زمان‌بند.

    ورودی‌ها:
        platform: شیء پلتفرم شامل opt (تنظیمات)، مسیرها و logger.
        model (torch.nn.Module): مدل هدف.
        epochs (int): تعداد epochهای آموزش.
        batch_size (int): اندازه‌ی مینی‌بتچ.
        mode (str): حالت اجرا ('pre' برای پیش‌آموزش، 'post' برای آموزش پس از هرس).
    """
    # ساخت DataLoaderهای آموزش/اعتبارسنجی و تست
    train_loader, val_loader = get_dataloader(
        platform.opt.dataset, batch_size,
        platform.opt.validation_size, platform.opt.dataloader_seed,
        loader_type="train_val"
    )
    test_loader = get_dataloader(platform.opt.dataset, batch_size, loader_type="test")

    # اجزای آموزش: معیار، بهینه‌ساز، زمان‌بند و تستر
    criterion = get_lossfunction()
    optimizer = get_optimizer(platform.opt, model, mode=mode)
    scheduler = get_scheduler(platform.opt, optimizer, mode=mode)
    tester = Tester(platform, criterion, val_loader, test_loader)

    # ارزیابی اولیه روی validation برای ثبت خط پایه
    tester.run(model, test_mode="val_eval")

    # شروع حلقه‌ی آموزش
    platform.logger.logger.info(f"\n{mode}-Training start!")
    for epoch in tqdm(range(epochs)):
        # یک epoch آموزش
        training_loss = train(model, platform.device, train_loader, optimizer, criterion)

        # ارزیابی روی val و test + ثبت مقادیر
        tester.run(model, training_loss=training_loss, epoch=epoch + 1, test_mode="val_eval")
        tester.run(model, test_mode="test_eval")

        # به‌روزرسانی زمان‌بند
        scheduler.step()

    # بارگذاری بهترین وزن اعتبارسنجی و تست نهایی (مطابق منطق موجود)
    if mode == 'pre' and len(val_loader) is not 0:
        model.load_state_dict(torch.load(tester.val_path))
        os.remove(tester.val_path)
    tester.run(model, test_mode="test_eval")

    # چاپ جدول و خلاصه‌ی نتایج
    tester.logging_table(platform.logger)
    platform.logger.logger.info(
        f"Training done!\n"
        f"Best validation accuracy: {tester.best_val_accu} (epoch: {tester.best_val_epoch})\n"
        f"Test accuracy: {model.train_testset_accuracy_arr.pop():.4f}\n"
        f"Test loss: {model.train_testset_loss_arr.pop():.8f}\n"
    )

    # ذخیره‌ی آرایه‌های نتایج (pre/post) و ریست آن‌ها برای استفاده‌های بعدی
    if mode == 'pre':
        save_result_data(model, platform.model_path['result_path'], mode="pre")
    elif mode == 'post':
        save_result_data(model, platform.model_path['result_path'], mode="post")
    result_data_initializer(model)

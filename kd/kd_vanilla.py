import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.data_utils import get_dataloader
from tqdm import tqdm
from training.train_utils import get_optimizer, get_scheduler
from training.Tester import Tester
from utils.base_utils import save_result_data, result_data_initializer


def train_kd(model, teacher_model, platform, train_loader, optimizer, criterion_kd):
    """
    حلقه‌ی آموزش KD:
    - معلم در حالت eval است و با no_grad خروجی می‌دهد.
    - اتلاف KD روی خروجی دانش‌آموز/معلم و لیبل‌ها محاسبه می‌شود.
    """
    model.train()
    teacher_model.eval()
    running_loss = 0.0

    for data, label in train_loader:
        data = data.to(platform.device)
        label = label.to(platform.device)

        optimizer.zero_grad()

        # پیش‌روی دانش‌آموز
        outputs = model(data)

        # پیش‌روی معلم بدون گرادیان
        with torch.no_grad():
            teacher_outputs = teacher_model(data)

        # محاسبه‌ی اتلاف KD و به‌روزرسانی وزن‌ها
        loss = criterion_kd(outputs, label, teacher_outputs, platform.opt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # میانگین اتلافِ هر مینی‌بتچ
    avg_loss = running_loss / len(train_loader)
    model.train_trainset_loss_arr.append(avg_loss)
    return avg_loss


# تابع اتلاف KD (vanilla) برگرفته از پیاده‌سازی مرجع
def loss_fn_kd(outputs, labels, teacher_outputs, opt):
    """
    KL(student||teacher) با دمای t و وزن alpha + CrossEntropy با وزن (1-alpha)
    """
    alpha = opt.kd_alpha
    t = opt.kd_temp

    kd_term = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs / t, dim=1),
        F.softmax(teacher_outputs / t, dim=1)
    ) * (alpha * t * t)

    ce_term = F.cross_entropy(outputs, labels) * (1.0 - alpha)
    return kd_term + ce_term


def run(platform, model, teacher_model):
    """
    اجرای حلقه‌ی KD به‌صورت vanilla: آموزش → اعتبارسنجی → انتخاب بهترین → تست.
    """
    # DataLoaderهای آموزش/اعتبارسنجی و تست
    train_loader, val_loader = get_dataloader(
        platform.opt.dataset,
        platform.opt.kd_batch_size,
        platform.opt.validation_size,
        platform.opt.dataloader_seed,
        loader_type="train_val"
    )
    test_loader = get_dataloader(
        platform.opt.dataset,
        platform.opt.kd_batch_size,
        loader_type="test"
    )

    # آماده‌سازی بهینه‌ساز/زمان‌بند و معیارها
    optimizer = get_optimizer(platform.opt, model, mode='kd')
    scheduler = get_scheduler(platform.opt, optimizer, mode='kd')
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = loss_fn_kd
    tester = Tester(platform, criterion_cls, val_loader, test_loader)

    # ارزیابی اولیه روی val
    tester.run(model, test_mode='val_eval')

    # حلقه‌ی آموزشی KD
    for epoch in tqdm(range(platform.opt.kd_epochs)):
        train_loss = train_kd(model, teacher_model, platform, train_loader, optimizer, criterion_kd)
        tester.run(model, training_loss=train_loss, epoch=epoch + 1, test_mode="val_eval")
        scheduler.step()

    # اگر مجموعه‌ی اعتبارسنجی داریم: بهترین وزنِ ذخیره‌شده را لود کن و فایل موقت را حذف کن
    if len(val_loader) != 0:
        model.load_state_dict(torch.load(tester.val_path))
        os.remove(tester.val_path)

    # ارزیابی نهایی روی test + گزارش
    tester.run(model, test_mode="test_eval")
    tester.logging_table(platform.logger)
    platform.logger.logger.info(
        "Training done!\n"
        f"Best validation accuracy: {tester.best_val_accu} (epoch: {tester.best_val_epoch})\n"
        f"Test accuracy: {model.train_testset_accuracy_arr.pop():.4f}\n"
        f"Test loss: {model.train_testset_loss_arr.pop():.8f}\n"
    )

    # ذخیره‌ی نتایج
    torch.save(model.state_dict(), os.path.join(platform.model_path["result_path"], "kd_model.pth"))
    save_result_data(model, platform.model_path['result_path'], mode="kd")
    result_data_initializer(model)

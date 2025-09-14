import os
import torch
import torch.nn as nn
from prettytable import PrettyTable


class Tester:
    """
    کلاس ارزیابی/لاگ‌گیری:
      - اجرای ارزیابی روی val/test
      - نگه‌داری بهترین دقتِ اعتبارسنجی و ذخیره‌ی وزن‌ها
      - ساخت و لاگ جدول نتایج هر epoch
    """
    def __init__(self, platform, criterion, val_loader, test_loader):
        self.criterion = criterion
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.platform = platform

        # جدول نتایج (PrettyTable)
        self.table = None
        self.table_initialize()

        # پیگیری بهترین دقتِ اعتبارسنجی
        self.best_val_accu = 0
        self.best_val_epoch = 0

        # مسیر ذخیره‌ی موقت بهترین وزن‌های val
        self.val_path = os.path.join(platform.model_path["result_path"], "val_weight_temp.pth")

        # اگر val_loader خالی بود، از test_loader برای val_eval استفاده می‌شود
        if len(val_loader) == 0:
            self.save_val = False
            self.val_loader = self.test_loader
        else:
            self.save_val = True

    def test(self, model, dataloader):
        """
        ارزیابی عمومی مدل روی یک DataLoader:
          - محاسبه‌ی loss تجمیعی و دقت
        """
        model.eval()
        correct = 0
        test_loss = 0
        total = 0

        with torch.no_grad():
            for data, label in dataloader:
                data = data.to(self.platform.device)
                label = label.to(self.platform.device)

                outputs = model(data)
                test_loss += self.criterion(outputs, label).item()
                correct += torch.sum(torch.eq(label, outputs.argmax(dim=1))).item()
                total += label.size(0)

        # محاسبه‌ی میانگین loss بر حسب تعداد نمونه‌ها و دقت
        test_loss /= total
        accuracy = correct / total
        return accuracy, test_loss

    def logging_per_epoch(self, accuracy, test_loss, training_loss, epoch):
        """
        افزودن یک سطر به جدول: epoch، اتلاف آموزش، اتلاف ارزیابی و دقت.
        """
        self.table.add_row([
            f"{epoch}",
            f"{training_loss:.8f}" if isinstance(training_loss, float) else "nan",
            f"{test_loss:.8f}",
            f"{accuracy:.4f}"
        ])

    def run(self, model, training_loss=0, epoch=0, test_mode=None):
        """
        اجرای ارزیابی و به‌روزرسانی لاگ/آرایه‌های ثبت:
          - test_mode='val_eval' → ارزیابی روی val، ذخیره‌ی بهترین، ثبت در جدول
          - test_mode='test_eval' → ارزیابی روی test، ثبت آرایه‌های مدل
        """
        if test_mode == "val_eval":
            accuracy, loss = self.test(model, self.val_loader)
            if self.save_val is True:
                self.validation_check(model, accuracy, epoch)
            model.val_accuracy_arr.append(accuracy)
            model.val_loss_arr.append(loss)
            self.logging_per_epoch(accuracy, loss, training_loss, epoch)

        elif test_mode == "test_eval":
            accuracy, loss = self.test(model, self.test_loader)
            model.train_testset_accuracy_arr.append(accuracy)
            model.train_testset_loss_arr.append(loss)

        else:
            raise ValueError("Please check test_mode. you can use test_eval or val_eval")

    def validation_check(self, model, accuracy, epoch):
        """
        مقایسه‌ی دقت فعلی val با بهترین مقدار؛ در صورت بهتر بودن، ذخیره‌ی وزن‌ها.
        """
        if accuracy > self.best_val_accu:
            self.best_val_accu = accuracy
            self.best_val_epoch = epoch
            torch.save(model.state_dict(), self.val_path)

    def table_initialize(self):
        """
        آماده‌سازی جدول PrettyTable برای گزارش دوره‌ای.
        """
        self.table = PrettyTable(['Epochs', 'Training_loss', 'Test_loss', 'Accuracy(val)'])
        self.table.align["Epochs"] = "l"
        self.table.align["Training_loss"] = "l"
        self.table.align["Test_loss"] = "l"
        self.table.align["Accuracy(val)"] = "l"

    def logging_table(self, logger):
        """
        لاگ‌کردن جدول نتایج با logger بیرونی.
        """
        logger.logger.info(self.table)

import glob
import os
from shutil import move
from torchvision import datasets


def TINYIMAGENET(root, train=True, transform=None, target_transform=None, download=False):
    """
    دانلود و آماده‌سازی دیتاست Tiny-ImageNet (200 کلاس).
    - اگر train=True: مسیر train بازگردانده می‌شود.
    - اگر train=False: مسیر val بازگردانده می‌شود (و در اولین دانلود ساختار آن اصلاح می‌گردد).
    """

    def _exists(root, filename):
        """بررسی می‌کند که آیا فایل داده در مسیر root وجود دارد یا خیر"""
        return os.path.exists(os.path.join(root, filename))

    def _download(url, root, filename):
        """دانلود و استخراج آرشیو دیتاست"""
        datasets.utils.download_and_extract_archive(
            url=url,
            download_root=root,
            extract_root=root,
            filename=filename
        )

    def _setup(root, base_folder):
        """
        تغییر ساختار پوشه‌ی val:
        - فایل val_annotations.txt را می‌خواند.
        - تصاویر val را به پوشه‌های کلاس‌های صحیح منتقل می‌کند.
        """

        target_folder = os.path.join(root, base_folder, 'val/')

        # خواندن لیست انوتیشن‌ها و ساخت دیکشنری {نام‌فایل: نام‌کلاس}
        val_dict = {}
        with open(os.path.join(target_folder, 'val_annotations.txt'), 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]

        # ایجاد پوشه برای هر کلاس (در صورت نبودن)
        paths = glob.glob(os.path.join(target_folder, 'images/*'))
        for path in paths:
            file = os.path.basename(path)
            folder = val_dict[file]
            class_dir = os.path.join(target_folder, folder)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

        # انتقال هر تصویر به پوشه‌ی کلاس خودش
        for path in paths:
            file = os.path.basename(path)
            folder = val_dict[file]
            dest = os.path.join(target_folder, folder, file)
            move(path, dest)

        # حذف فایل انوتیشن و پوشه‌ی images
        os.remove(os.path.join(target_folder, 'val_annotations.txt'))
        os.rmdir(os.path.join(target_folder, 'images'))

    # لینک رسمی دیتاست Tiny-ImageNet
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'

    # اگر دانلود فعال باشد و فایل وجود نداشته باشد → دانلود و آماده‌سازی
    if download and not _exists(root, filename):
        _download(url, root, filename)
        _setup(root, base_folder)

    # انتخاب پوشه‌ی train یا val
    folder = os.path.join(root, base_folder, 'train' if train else 'val')

    # برگرداندن دیتاست به فرمت ImageFolder
    return datasets.ImageFolder(
        folder,
        transform=transform,
        target_transform=target_transform
    )

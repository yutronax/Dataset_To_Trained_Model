import random
import gradio as gr
import pandas as pd
import zipfile
import os
import shutil
import uuid
class model_secim_duzenleme:
    def __init__(self):
        self.model_secim=None
        self.dosya_yolu = None
        self.class_names = None
        self.class_nums = None


    def dosya_Zipmi(self, zip_file):
        zip_path = zip_file
        if not zipfile.is_zipfile(zip_path):
            return zip_path

        session_id = str(uuid.uuid4())
        extract_to = os.path.join("extracted_data", session_id)
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)


        return  extract_to

    def model_secim_belirle(self, path):
        directories = []
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                directories.append(item)

        if len(directories) == 2:
            all_labels = True
            for folder in directories:
                folder_path = os.path.join(path, folder)
                for f in os.listdir(folder_path):
                    if not any(f.lower().endswith(ext) for ext in ['txt', 'xml', 'json']):
                        all_labels = False
                        break
                if not all_labels:
                    break

            if all_labels:
                return "Nesne Tespiti"

            lower_dirs = [d.lower() for d in directories]
            if ('masks' in lower_dirs or 'mask' in lower_dirs) and len(directories) == 2:
                folder1_files = os.listdir(os.path.join(path, directories[0]))
                folder2_files = os.listdir(os.path.join(path, directories[1]))
                if len(folder1_files) == len(folder2_files):
                    return "Semantik Segmentasyon"

        return "Nesne Sınıflandırma"

    def is_already_split(self, dataset_path):
        alt_klasorler = os.listdir(dataset_path)
        split_kelimeleri = ['train', 'test', 'val']
        bulunanlar = [klasor for klasor in alt_klasorler if klasor.lower() in split_kelimeleri]
        return len(bulunanlar) >= 2

    def klasor_olustur(self, *paths):
        for path in paths:
            os.makedirs(path, exist_ok=True)

    def dosyalari_kopyala_karsilikli(self, src_files, dest_dir):
        for src_path in src_files:
            dosya_adi = os.path.basename(src_path)
            if os.path.isfile(src_path):
              shutil.copy2(src_path, os.path.join(dest_dir, dosya_adi))

    def split_siniflandirma(self, dataset_path, test_orani=0.2):
        siniflar = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if len(siniflar) == 1:
            new_dataset =os.path.join(dataset_path, siniflar[0])
            siniflar = [d for d in os.listdir(new_dataset) if os.path.isdir(os.path.join(new_dataset, d))]
            dataset_path = new_dataset

        for sinif in siniflar:
            klasor = os.path.join(dataset_path, sinif)
            dosyalar = [os.path.join(klasor, f) for f in os.listdir(klasor)]
            random.shuffle(dosyalar)

            test_sayisi = int(len(dosyalar) * test_orani)
            test_dosyalar = dosyalar[:test_sayisi]
            train_dosyalar = dosyalar[test_sayisi:]

            self.klasor_olustur(
                os.path.join(r"/content/trainer", "train", sinif),
                os.path.join(r"/content/trainer", "test", sinif)
            )

            self.dosyalari_kopyala_karsilikli(train_dosyalar,os.path.join(r"/content/trainer", "train", sinif))
            self.dosyalari_kopyala_karsilikli(test_dosyalar, os.path.join(r"/content/trainer", "test", sinif))

    def split_segmentasyon(self, dataset_path, test_orani=0.2):
        images_dir = [d for d in os.listdir(dataset_path) if "masks" not in d.lower()][0]
        masks_dir = [d for d in os.listdir(dataset_path) if "masks" in d.lower()][0]

        images_path = os.path.join(dataset_path, images_dir)
        masks_path = os.path.join(dataset_path, masks_dir)

        img_list = sorted(os.listdir(images_path))
        test_sayisi = int(len(img_list) * test_orani)
        random.shuffle(img_list)

        test_imgs = img_list[:test_sayisi]
        train_imgs = img_list[test_sayisi:]

        for alt in ['train/images', 'train/masks', 'test/images', 'test/masks']:
            self.klasor_olustur(os.path.join(dataset_path, alt))

        for img in train_imgs:
            shutil.copy2(os.path.join(images_path, img), os.path.join(dataset_path, 'train/images', img))
            shutil.copy2(os.path.join(masks_path, img), os.path.join(dataset_path, 'train/masks', img))

        for img in test_imgs:
            shutil.copy2(os.path.join(images_path, img), os.path.join(dataset_path, 'test/images', img))
            shutil.copy2(os.path.join(masks_path, img), os.path.join(dataset_path, 'test/masks', img))

    def split_nesne_tespiti(self, dataset_path, test_orani=0.2):
        klasorler = os.listdir(dataset_path)
        img_klasor = [k for k in klasorler if 'image' in k.lower()][0]
        label_klasor = [k for k in klasorler if k != img_klasor][0]

        img_path = os.path.join(dataset_path, img_klasor)
        label_path = os.path.join(dataset_path, label_klasor)

        img_list = sorted(os.listdir(img_path))
        test_sayisi = int(len(img_list) * test_orani)
        random.shuffle(img_list)

        test_imgs = img_list[:test_sayisi]
        train_imgs = img_list[test_sayisi:]

        for alt in ['train/images', 'train/labels', 'test/images', 'test/labels']:
            self.klasor_olustur(os.path.join(dataset_path, alt))

        for img in train_imgs:
            label_name = os.path.splitext(img)[0] + ".txt"
            shutil.copy2(os.path.join(img_path, img), os.path.join(dataset_path, 'train/images', img))
            shutil.copy2(os.path.join(label_path, label_name), os.path.join(dataset_path, 'train/labels', label_name))

        for img in test_imgs:
            label_name = os.path.splitext(img)[0] + ".txt"
            shutil.copy2(os.path.join(img_path, img), os.path.join(dataset_path, 'test/images', img))
            shutil.copy2(os.path.join(label_path, label_name), os.path.join(dataset_path, 'test/labels', label_name))

    def otomatik_split_et(self, dataset_path):
        if self.is_already_split(dataset_path):
            shutil.move(dataset_path,r"/content/trainer")
            print("Dataset zaten split edilmiş!")
            return

        self.model_secim = self.model_secim_belirle(dataset_path)
        print(f"Tespit edilen görev: {self.model_secim}")

        if self.model_secim == "Nesne Sınıflandırma":
            self.split_siniflandirma(dataset_path)
        elif self.model_secim == "Semantik Segmentasyon":
            self.split_segmentasyon(dataset_path)
        elif self.model_secim == "Nesne Tespiti":
            self.split_nesne_tespiti(dataset_path)

        print("Split işlemi tamamlandı!")


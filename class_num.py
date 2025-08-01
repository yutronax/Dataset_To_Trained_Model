
class SinifSayisiBulucu:
    def __init__(self):
        pass

    def sinif_sayisi_classification(self, dataset_path):
        train_path = os.path.join(dataset_path, "train")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"{train_path} bulunamadı.")
        siniflar = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        return len(siniflar)

    def sinif_sayisi_segmentation(self, mask_folder_path):
        if not os.path.exists(mask_folder_path):
            raise FileNotFoundError(f"{mask_folder_path} bulunamadı.")
        max_label = 0
        mask_files = [f for f in os.listdir(mask_folder_path) if f.lower().endswith((".png", ".jpg", ".tif"))]
        for mask_file in mask_files:
            mask_path = os.path.join(mask_folder_path, mask_file)
            mask = np.array(Image.open(mask_path))
            max_label = max(max_label, mask.max())
        return max_label + 1  # 0'dan başlıyor

    def sinif_sayisi_detection(self, label_folder_path):
        if not os.path.exists(label_folder_path):
            raise FileNotFoundError(f"{label_folder_path} bulunamadı.")
        max_class_id = -1
        label_files = [f for f in os.listdir(label_folder_path) if f.endswith(".txt")]
        for label_file in label_files:
            with open(os.path.join(label_folder_path, label_file), "r") as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if class_id > max_class_id:
                            max_class_id = class_id
        return max_class_id + 1

    def sinif_sayisi_ogren(self, dataset_path, gorev):
        if gorev == "Nesne Sınıflandırma":
            return self.sinif_sayisi_classification(dataset_path)
        elif gorev == "Semantik Segmentasyon":
            # mask klasörünü otomatik bul
            mask_folder = None
            for d in os.listdir(dataset_path):
                if "mask" in d.lower():
                    mask_folder = os.path.join(dataset_path, d)
                    break
            if mask_folder is None:
                raise FileNotFoundError("Mask klasörü bulunamadı.")
            return self.sinif_sayisi_segmentation(mask_folder)
        elif gorev == "Nesne Tespiti":
            # label klasörünü otomatik bul
            label_folder = None
            for d in os.listdir(dataset_path):
                if d.lower() not in ["images", "image", "img", "imgs"]:
                    # labels klasörü olduğunu varsay
                    label_folder = os.path.join(dataset_path, d)
                    break
            if label_folder is None:
                raise FileNotFoundError("Label klasörü bulunamadı.")
            return self.sinif_sayisi_detection(label_folder)
        else:
            raise ValueError("Geçersiz görev tipi.")


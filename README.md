# Dataset_To_Trained_Model

## 📋 Proje Özeti / Project Summary

**TR:** Bu proje, herhangi bir veri seti yükleyerek otomatik olarak makine öğrenmesi modeli eğiten kullanıcı dostu bir sistemdir. Gradio tabanlı web arayüzü ile görüntü sınıflandırma, semantik segmentasyon ve nesne tespiti görevleri için model eğitebilirsiniz.

**EN:** This project is a user-friendly system that automatically trains machine learning models by uploading any dataset. With a Gradio-based web interface, you can train models for image classification, semantic segmentation, and object detection tasks.

---

## 🎯 Özellikler / Features

### Otomatik Model Tipi Tespiti / Automatic Model Type Detection
- 📊 **Nesne Sınıflandırma** / Image Classification (ResNet18)
- 🎨 **Semantik Segmentasyon** / Semantic Segmentation (U-Net)
- 🔍 **Nesne Tespiti** / Object Detection (Faster R-CNN)

### Akıllı Veri İşleme / Intelligent Data Processing
- ✅ Otomatik veri seti formatı tanıma
- ✅ Otomatik train/test split (80/20)
- ✅ ZIP dosyası desteği
- ✅ Çoklu sınıf desteği

### Eğitim Özellikleri / Training Features
- ⚙️ Ayarlanabilir hiperparametreler (epoch, batch size, learning rate)
- 📈 Early stopping mekanizması
- 💾 En iyi model otomatik kaydetme
- 🎯 Validation set ile model değerlendirme
- 📊 Test sonuçları ve doğruluk metrikleri

---

## 🏗️ Proje Yapısı / Project Structure

```
Dataset_To_Trained_Model/
├── Interface.py                    # Gradio web arayüzü / Web interface
├── model_training.py               # Model eğitim sınıfları / Training classes
├── model_selection_and_split.py    # Veri hazırlama / Data preparation
├── class_num.py                    # Sınıf sayısı tespit / Class count detection
├── Tester.py                       # Model test ve değerlendirme / Testing
└── README.md                       # Proje dokümantasyonu
```

---

## 🚀 Kurulum / Installation

### Gereksinimler / Requirements

```bash
pip install torch torchvision gradio pillow matplotlib numpy
```

### Çalıştırma / Running

```bash
python Interface.py
```

Tarayıcınızda otomatik olarak açılacak arayüzden sistemi kullanabilirsiniz.

---

## 📖 Kullanım / Usage

### 1. Veri Seti Hazırlama / Dataset Preparation

#### Nesne Sınıflandırma / Image Classification
```
dataset.zip
└── class1/
    ├── image1.jpg
    ├── image2.jpg
└── class2/
    ├── image1.jpg
    ├── image2.jpg
```

#### Semantik Segmentasyon / Semantic Segmentation
```
dataset.zip
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
└── masks/
    ├── img1.jpg
    ├── img2.jpg
```

#### Nesne Tespiti / Object Detection
```
dataset.zip
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
└── labels/
    ├── img1.txt
    ├── img2.txt
```

### 2. Model Eğitimi / Model Training

1. ZIP dosyasını yükleyin
2. Hiperparametreleri ayarlayın:
   - **Epoch Sayısı** (5-100): Kaç kez veri seti üzerinden geçilecek
   - **Batch Size** (4-32): Her iterasyonda işlenecek örnek sayısı
   - **Learning Rate** (0.0001-0.01): Öğrenme hızı
3. "Eğitimi Başlat" butonuna tıklayın
4. Sonuçları görüntüleyin ve modeli indirin

---

## 🔬 Teknik Detaylar / Technical Details

### Desteklenen Modeller / Supported Models

1. **ResNet18** (Classification)
   - Transfer learning with pretrained weights
   - Custom fully connected layer for class count
   - Cross-entropy loss

2. **U-Net** (Segmentation)
   - Encoder-decoder architecture
   - Skip connections
   - Binary cross-entropy loss
   - IoU metric

3. **Faster R-CNN** (Detection)
   - ResNet50-FPN backbone
   - Region Proposal Network
   - Multi-task loss

### Eğitim Stratejisi / Training Strategy

- **Validation Split**: %10 (from training data)
- **Early Stopping**: 5 epochs patience
- **Optimizer**: Adam
- **Device**: Automatic GPU/CPU detection

---

## 📊 Çıktılar / Outputs

Sistem aşağıdaki çıktıları sağlar:

1. **Eğitim Sonuçları**: Model tipi, sınıf sayısı, epoch, batch size, learning rate
2. **Eğitilen Model**: `.pth` formatında PyTorch model dosyası
3. **Doğruluk Metrikleri**: 
   - Sınıflandırma için accuracy (%)
   - Segmentasyon için IoU (%)
   - Nesne tespiti için detection confidence (%)
4. **Test Görseli**: Model tahminlerinin görsel örneği

---

## 🎓 Kullanım Senaryoları / Use Cases

- 🏥 **Medikal Görüntü Analizi**: Tümör tespiti, organ segmentasyonu
- 🚗 **Otonom Araçlar**: Nesne tespiti, yol segmentasyonu
- 🌾 **Tarım**: Bitki hastalık sınıflandırması
- 🏭 **Endüstri**: Kalite kontrol, defekt tespiti
- 🐾 **Hayvan Tanıma**: Tür sınıflandırması
- 📸 **Fotoğraf Analizi**: Nesne tanıma ve segmentasyon

---

## 🛠️ Teknoloji Stack / Technology Stack

- **Deep Learning**: PyTorch, TorchVision
- **Web Interface**: Gradio
- **Image Processing**: PIL, Matplotlib
- **Data Processing**: NumPy
- **Model Architectures**: ResNet18, U-Net, Faster R-CNN

---

## 📝 Notlar / Notes

- Sistem otomatik olarak GPU kullanır (varsa)
- Tüm modeller PyTorch formatında kaydedilir
- Early stopping ile aşırı öğrenme önlenir
- Test sonuçları görselleştirilir

---

## 🔮 Gelecek Geliştirmeler / Future Improvements

- [ ] Daha fazla model mimarisi desteği (EfficientNet, YOLO, etc.)
- [ ] Data augmentation seçenekleri
- [ ] TensorBoard entegrasyonu
- [ ] Model karşılaştırma özelliği
- [ ] ONNX export desteği
- [ ] Batch prediction özelliği

---

## 👨‍💻 Geliştirici / Developer

**Yutronax**

Bu proje, makine öğrenmesi modellerini kolayca eğitmek isteyen herkes için geliştirilmiştir. Herhangi bir sorunuz veya öneriniz varsa lütfen iletişime geçin.

---

## 📄 Lisans / License

Bu proje açık kaynak kodludur ve eğitim amaçlı kullanılabilir.
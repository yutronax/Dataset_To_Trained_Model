import gradio as gr
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tempfile
import shutil
from torchvision import transforms, models
import torch.nn as nn
def train_model(zip_file, epochs, batch_size, learning_rate):
    try:
        temp_dir = tempfile.mkdtemp()

        data = model_secim_duzenleme()
        trainer = Egitici()
        class_finder = SinifSayisiBulucu()
        tester = Tester()

        extracted_path = data.dosya_Zipmi(zip_file.name)
        data.otomatik_split_et(extracted_path)

        model_type = data.model_secim
        train_path = "/content/trainer"
        class_count = class_finder.sinif_sayisi_ogren(train_path, model_type)

        if model_type == "Nesne Sınıflandırma":
            trainer.egit_siniflandirma(train_path, epochs=epochs, batch_size=batch_size, lr=learning_rate)
            model_file = os.path.join(train_path, "en_iyi_model_sinif.pth")
            class_names = sorted(os.listdir(os.path.join(train_path, "train")))
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
            model.load_state_dict(torch.load(model_file))
            accuracy, test_image = tester.test_siniflandirma(os.path.join(train_path, "test"), model, class_names)

        elif model_type == "Semantik Segmentasyon":
            trainer.egit_segmentasyon(train_path, epochs=epochs, batch_size=batch_size, lr=learning_rate)
            model_file = "en_iyi_unet_model.pth"
            model = UNet(in_channels=3, out_channels=1)
            model.load_state_dict(torch.load(model_file))
            test_image = tester.test_segmentasyon(os.path.join(train_path, "test"), model)

        elif model_type == "Nesne Tespiti":
            model = Egitici.get_detection_model(class_count)
            trainer.egit_nesne_tespiti(model, train_path, epochs=epochs, batch_size=batch_size, lr=learning_rate)
            model_file = "en_iyi_nesne_tespit_model.pth"
            model.load_state_dict(torch.load(model_file))
            test_image = tester.test_tespit(model, os.path.join(train_path, "test"))

        results = f"Model Tipi: {model_type}\nSınıf Sayısı: {class_count}\nEpoch: {epochs}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nAccuracy: %{accuracy:.2f}"

        return results, model_file, class_count, f"%{accuracy:.2f}", test_image

    except Exception as e:
        return f"Hata: {str(e)}", None, 0, "0%", None




def create_interface():
    with gr.Blocks(title="ML Model Eğitim Sistemi", theme=gr.themes.Soft()) as iface:

        gr.Markdown("# ML Model Eğitim Sistemi")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Giriş")

                zip_input = gr.File(
                    label="Dataset ZIP Dosyası",
                    file_types=[".zip"],
                    type="filepath"
                )

                with gr.Row():
                    epochs = gr.Slider(
                        minimum=5, maximum=100, value=30, step=5,
                        label="Epoch Sayısı"
                    )
                    batch_size = gr.Slider(
                        minimum=4, maximum=32, value=16, step=4,
                        label="Batch Size"
                    )

                learning_rate = gr.Slider(
                    minimum=0.0001, maximum=0.01, value=0.001, step=0.0001,
                    label="Learning Rate"
                )

                train_btn = gr.Button("Eğitimi Başlat", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("## Çıkış")

                result_text = gr.Textbox(
                    label="Eğitim Sonuçları",
                    lines=10,
                    max_lines=15
                )

                with gr.Row():
                    model_file = gr.File(label="Eğitilen Model")
                    class_count = gr.Number(label="Sınıf Sayısı", precision=0,interactive=False)

                with gr.Row():
                    accuracy = gr.Textbox(label="Model Doğruluğu")


                test_image = gr.Image(label="Test Sonucu Görseli")

        train_btn.click(
            fn=train_model,
            inputs=[zip_input, epochs, batch_size, learning_rate],
            outputs=[result_text, model_file, class_count, accuracy, test_image]
        )

    return iface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0")

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms

class Tester:
  @staticmethod
  def test_siniflandirma(test_path, model, class_names):
      model.eval()
      total = 0
      correct = 0
      first_img_to_show = None  # ilk resmi döndürmek için

      for class_name in class_names:
          class_path = os.path.join(test_path, class_name)
          if not os.path.isdir(class_path):
              continue
          test_images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]
          with torch.no_grad():
              for i, img_path in enumerate(test_images):
                  image = Image.open(img_path).convert("RGB")
                  transform = transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.ToTensor()
                  ])
                  input_tensor = transform(image).unsqueeze(0)
                  output = model(input_tensor)
                  pred = torch.argmax(output, dim=1).item()

                  if pred == class_names.index(class_name):
                      correct += 1
                  total += 1

                  # ilk görseli test_image olarak döndür
                  if first_img_to_show is None and i == 0:
                      fig, ax = plt.subplots()
                      ax.imshow(image)
                      ax.set_title(f"Tahmin: {class_names[pred]}, Gerçek: {class_name}")
                      ax.axis("off")
                      from io import BytesIO
                      buf = BytesIO()
                      plt.savefig(buf, format='png')
                      buf.seek(0)
                      first_img_to_show = Image.open(buf)

      accuracy = correct / total * 100 if total > 0 else 0
      return accuracy, first_img_to_show



  @staticmethod
  def test_segmentasyon(test_path, model):
    from torchvision import transforms
    model.eval()
    images_path = os.path.join(test_path, "images")
    masks_path = os.path.join(test_path, "masks")
    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        print("Images or Masks folder not found")
        return "0%", None

    test_images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    iou_scores = []
    image_for_display = None

    with torch.no_grad():
        for i, img_name in enumerate(test_images):
            img_path = os.path.join(images_path, img_name)
            mask_path = os.path.join(masks_path, img_name)

            image = Image.open(img_path).convert("RGB")
            gt_mask = Image.open(mask_path).convert("L")

            input_tensor = transform(image).unsqueeze(0)
            output = model(input_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

            gt_mask = transform(gt_mask).squeeze().numpy()
            gt_mask = (gt_mask > 0.5).astype(np.uint8)

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)

            if i == 0:
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                axs[0].imshow(image)
                axs[0].set_title("Girdi Görseli")
                axs[0].axis("off")
                axs[1].imshow(pred_mask, cmap="gray")
                axs[1].set_title("Tahmin Maske")
                axs[1].axis("off")

                from io import BytesIO
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image_for_display = Image.open(buf)

    avg_iou = np.mean(iou_scores) if iou_scores else 0
    return f"%{avg_iou*100:.2f}", image_for_display




  @staticmethod
  def test_tespit(model, test_path, threshold=0.5):
    from torchvision.transforms import functional as F
    model.eval()
    images_path = os.path.join(test_path, "images")
    if not os.path.exists(images_path):
        print("Images path not found")
        return "0%", None

    test_images = [os.path.join(images_path, img) for img in os.listdir(images_path) if img.endswith(('.jpg', '.png'))]
    score_list = []
    display_image = None

    for idx, img_path in enumerate(test_images[:5]):
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        scores = prediction["scores"].cpu().numpy()
        score_list.extend(scores[scores > threshold])

        if idx == 0:
            fig, ax = plt.subplots()
            ax.imshow(image)
            for box, score in zip(prediction["boxes"], prediction["scores"]):
                if score > threshold:
                    x1, y1, x2, y2 = box
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, color="red", linewidth=2))
            ax.set_title("Tespit Sonucu")
            ax.axis("off")
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            display_image = Image.open(buf)

    avg_score = np.mean(score_list) if score_list else 0
    return f"%{avg_score*100:.2f}", display_image

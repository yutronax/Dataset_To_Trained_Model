


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
import numpy as np
from PIL import Image

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(self.double_conv(in_channels, feature))
            in_channels = feature

        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self.double_conv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
class Egitici:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def egit_siniflandirma(self, dataset_path, epochs=30, batch_size=16, lr=0.001, patience=5, val_orani=0.1):
        train_dir = os.path.join(dataset_path, "train")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        full_dataset = datasets.ImageFolder(train_dir, transform=transform)
        class_count = len(full_dataset.classes)

        val_size = int(len(full_dataset) * val_orani)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, class_count)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        early_stop_counter = 0

        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), os.path.join(dataset_path, "en_iyi_model_sinif.pth"))
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    break

    def egit_segmentasyon(self, dataset, epochs=30, batch_size=8, lr=1e-3,
                          val_orani=0.1, early_stop_patience=5):
        device = self.device
        model = UNet(in_channels=3, out_channels=1).to(device)

        val_size = int(len(dataset) * val_orani)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "en_iyi_unet_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

    def egit_nesne_tespiti(self, model, full_dataset, epochs=25, batch_size=8, lr=1e-4,
                           val_orani=0.1, early_stop_patience=5):
        device = self.device
        model.to(device)

        val_size = int(len(full_dataset) * val_orani)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda x: tuple(zip(*x)))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for images, targets in train_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    total_val_loss += losses.item()

            avg_val_loss = total_val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_save_path = os.path.join("en_iyi_nesne_tespit_model.pth")
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
    def get_detection_model(num_classes):

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model



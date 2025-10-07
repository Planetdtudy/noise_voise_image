# ----------------------------
# Imports
# ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
image_size = 224
tabular_dim = 5
hidden_dim = 128
num_classes = 2
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
number_of_images_clean=235
# ----------------------------
# Dataset for chest-xray folder
# Noisy class generated on-the-fly from clean images using add_noise
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # Optional: uncomment for Swin pretrained normalization
    # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.tabular = []
        self.labels = []

        # Assuming folder structure: chest-xray/clean/ and chest-xray/noisy/
        for label, subfolder in enumerate(['clean', 'noisy']):
            folder_path = os.path.join(root_dir, subfolder)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(folder_path, fname))
                    # Tabular features: zeros for clean, ones for noisy
                    self.tabular.append([0]*tabular_dim if label==0 else [1]*tabular_dim)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = transform(img)
        tabular = torch.tensor(self.tabular[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, tabular, label

# ----------------------------
# DataLoader
# ----------------------------
dataset = ChestXRayDataset('chest_xray')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# Swin Tiny CNN
# ----------------------------
cnn = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
cnn.head = nn.Identity()  # remove classifier
cnn.eval()  # freeze CNN

# ----------------------------
# Fusion MLP
# ----------------------------
class FusionModel(nn.Module):
    def __init__(self, image_embed_dim, tabular_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_embed_dim + tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, img_emb, tabular):
        x = torch.cat([img_emb, tabular], dim=1)
        return self.fc(x)

# Get embedding size from CNN
with torch.no_grad():
    sample_imgs, _, _ = next(iter(loader))
    sample_emb = cnn.forward_features(sample_imgs[:2])
    sample_emb = sample_emb.mean(dim=[2,3])

fusion_model = FusionModel(image_embed_dim=sample_emb.shape[1],
                           tabular_dim=tabular_dim,
                           hidden_dim=hidden_dim,
                           num_classes=num_classes)

# ----------------------------
# Training setup
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=learning_rate)
fusion_model.train()

# ----------------------------
# Training loop
# ----------------------------
total_images = len(dataset)

for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for imgs, tabular, lbls in loader:
        optimizer.zero_grad()

        # Step 1: CNN embeddings
        with torch.no_grad():  # freeze CNN
            img_emb = cnn.forward_features(imgs)
            img_emb = img_emb.mean(dim=[2,3])

        # Step 2: Fusion + forward
        outputs = fusion_model(img_emb, tabular)

        # Step 3: Loss
        loss = criterion(outputs, lbls)

        # Step 4: Backward
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == lbls).float().mean()

        running_loss += loss.item() * imgs.size(0)
        running_acc += acc.item() * imgs.size(0)

    epoch_loss = running_loss / total_images
    epoch_acc = running_acc / total_images
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.2f}%")

# ----------------------------
# Plot a few predictions
# ----------------------------
fusion_model.eval()
with torch.no_grad():
    sample_imgs, sample_tabular, sample_labels = next(iter(loader))
    img_emb = cnn.forward_features(sample_imgs)
    img_emb = img_emb.mean(dim=[2,3])
    outputs = fusion_model(img_emb, sample_tabular)
    preds = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(1, min(8, sample_imgs.shape[0]), figsize=(16,4))
    for i in range(min(8, sample_imgs.shape[0])):
        ax = axes[i] if sample_imgs.shape[0] > 1 else axes
        ax.imshow(sample_imgs[i].permute(1,2,0).numpy())
        ax.set_title(f"L:{sample_labels[i].item()} P:{preds[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

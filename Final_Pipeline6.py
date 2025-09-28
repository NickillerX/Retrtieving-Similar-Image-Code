import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# (Configuration and Model Classes are the same)
CONFIG = {
    "image_size": (224, 224), "num_classes": 20,
    "model_dims": {"customcnn_v1": 128, "customcnn_v2": 128, "resnet50": 2048, "efficientnet_b0": 1280, "vgg16": 4096}
}


class CustomCNN_v1(nn.Module):
    def __init__(self, num_classes=CONFIG["num_classes"]):
        super(CustomCNN_v1, self).__init__();
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2));
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2));
        self.conv_block3 = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.MaxPool2d(2, 2));
        self.conv_block4 = nn.Sequential(nn.Conv2d(256, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.ReLU(),
                                         nn.MaxPool2d(2, 2));
        self.flatten = nn.Flatten();
        self.fc1 = nn.Linear(1024 * 14 * 14, 1024);
        self.fc2 = nn.Linear(1024, 512);
        self.fc3 = nn.Linear(512, 128);
        self.fc_out = nn.Linear(128, num_classes);
        self.dropout = nn.Dropout(0.15)

    def forward(self, x, extract_features=False):
        x = self.conv_block1(x);
        x = self.conv_block2(x);
        x = self.conv_block3(x);
        x = self.conv_block4(x);
        x = self.flatten(x);
        x = F.relu(self.fc1(x));
        x = F.relu(self.fc2(x));
        x = self.dropout(x);
        features = F.relu(self.fc3(x))
        if extract_features: return features
        return self.fc_out(features)


class CustomCNN_v2(nn.Module):
    def __init__(self, num_classes=CONFIG["num_classes"]):
        super(CustomCNN_v2, self).__init__();
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2));
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2));
        self.conv_block3 = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                         nn.MaxPool2d(2, 2));
        self.conv_block4 = nn.Sequential(nn.Conv2d(256, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.ReLU(),
                                         nn.MaxPool2d(2, 2));
        self.pool = nn.AdaptiveAvgPool2d((1, 1));
        self.flatten = nn.Flatten();
        self.fc1 = nn.Linear(1024, 512);
        self.fc2 = nn.Linear(512, 128);
        self.fc_out = nn.Linear(128, num_classes);
        self.dropout = nn.Dropout(0.12)

    def forward(self, x, extract_features=False):
        x = self.conv_block1(x);
        x = self.conv_block2(x);
        x = self.conv_block3(x);
        x = self.conv_block4(x);
        x = self.pool(x);
        x = self.flatten(x);
        x = F.relu(self.fc1(x));
        x = self.dropout(x);
        features = F.relu(self.fc2(x))
        if extract_features: return features
        return self.fc_out(features)


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, model_path, device):
        super().__init__();
        self.model_name = model_name;
        self.device = device
        self.model = self._load_full_model(model_name, model_path).to(device).eval()

    def _load_full_model(self, model_name, model_path):
        if model_name == "customcnn_v1":
            model = CustomCNN_v1()
        elif model_name == "customcnn_v2":
            model = CustomCNN_v2()
        elif model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        if model_name == "resnet50":
            model.fc = nn.Linear(model.fc.in_features, CONFIG["num_classes"])
        elif model_name == "vgg16":
            model.classifier[6] = nn.Linear(4096, CONFIG["num_classes"])
        elif model_name == "efficientnet_b0":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, CONFIG["num_classes"])

        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device)); print(
                    f"Loaded weights: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"Error loading weights: {e}.")
        else:
            print(f"Warning: Weights not found at {model_path}.")
        return model

    def forward(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            if "customcnn" in self.model_name:
                features = self.model(x, extract_features=True)
            elif self.model_name == "resnet50":
                features = nn.Sequential(*list(self.model.children())[:-1])(x)
            elif self.model_name == "efficientnet_b0":
                features = nn.Sequential(*list(self.model.children())[:-1])(x)

            # --- MODIFIED: VGG16 logic now matches the Colab script exactly ---
            elif self.model_name == "vgg16":
                x = self.model.features(x)
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                # Manually pass through the first few classifier layers, stopping where the Colab script stopped
                x = self.model.classifier[0](x)
                x = F.relu(x, inplace=True)
                x = self.model.classifier[1](x)
                x = F.relu(x, inplace=True)
                features = self.model.classifier[2](x)  # Stop at classifier[2]

            else:
                raise ValueError(f"Feature extraction not defined for {self.model_name}")

            return torch.flatten(features, 1).cpu().numpy()


# (The rest of the helper functions remain the same)
def preprocess_image(path):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    try:
        return transform(Image.open(path).convert('RGB')).unsqueeze(0)
    except Exception:
        return None


def load_feature_vectors(path, dim, root):
    if not os.path.exists(path): return None, None, None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if not all(k in data for k in ['features', 'paths', 'labels']): return None, None, None
    features = np.array(data['features'])
    if features.ndim != 2 or features.shape[1] != dim: print(
        f"Feature dimension mismatch! Expected {dim}, got {features.shape[1]}");return None, None, None
    paths, labels = data['paths'], data['labels']
    try:
        prefix = os.path.commonpath(paths);corrected = [os.path.join(root, os.path.relpath(p, start=prefix)) for p in
                                                        paths]
    except(ValueError, TypeError):
        corrected = paths
    print(f"Loaded {len(features)} features from {os.path.basename(path)}");
    return features, corrected, labels


def search_similar(query_feat, features, paths, k=5, thresh=0.7):
    if features is None: return []
    sims = cosine_similarity(query_feat, features).flatten()
    indices = np.where(sims >= thresh)[0]
    if len(indices) == 0: return []
    top_indices = indices[np.argsort(sims[indices])[::-1][:k]]
    return [(paths[i], sims[i]) for i in top_indices]


def run_feature_extraction(extractor, dataset_path, save_path):
    image_paths = glob.glob(os.path.join(dataset_path, '**', '*.*'), recursive=True);
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_paths: return False, "No images found"
    features, paths, labels = [], [], [];
    start_time = time.time();
    tasks = [(extractor, path) for path in image_paths]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(_extract_single_image, tasks), total=len(tasks), desc="Extracting Features"))
    for result in results:
        if result: feature, path, label = result;features.append(feature);paths.append(path);labels.append(label)
    if not features: return False, "Failed to extract features"
    class_names = sorted(list(set(labels)));
    label_to_idx = {name: i for i, name in enumerate(class_names)};
    numeric_labels = [label_to_idx[lbl] for lbl in labels]
    with open(save_path, 'wb') as f:
        pickle.dump({'features': np.array(features), 'paths': paths, 'labels': numeric_labels}, f, protocol=4)
    duration = time.time() - start_time;
    return True, f"Saved {len(features)} features to {os.path.basename(save_path)} in {duration:.2f}s."


def _extract_single_image(args):
    extractor, img_path = args
    try:
        img_tensor = preprocess_image(img_path);feature_vector = extractor(
            img_tensor).squeeze().tolist();label_str = os.path.basename(os.path.dirname(img_path));return (
        feature_vector, img_path, label_str)
    except Exception as e:
        print(f"Error processing {img_path}: {e}");return None
"""
Neural Style Transfer

This script demonstrates:
1. Transferring artistic style from one image to content of another
2. Content loss and style loss (Gram matrices)
3. Perceptual loss using pre-trained VGG
4. Fast neural style transfer
5. Multiple style learning

Reference: "A Neural Algorithm of Artistic Style" (Gatys et al.)
Applications: Artistic filters, creative tools, video stylization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
CONFIG = {
    'image_size': 512,
    'style_weight': 1000000,
    'content_weight': 1,
    'tv_weight': 1e-6,  # Total variation loss weight
    'num_steps': 300,
    'learning_rate': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'results/style_transfer'
}

# VGG19 layers for style and content
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_LAYERS = ['conv4_2']

class VGGFeatureExtractor(nn.Module):
    """
    VGG19 for extracting style and content features

    Uses pre-trained VGG19 to extract intermediate layer activations
    """

    def __init__(self, device='cpu'):
        super().__init__()

        print("\n" + "="*80)
        print("LOADING VGG19 FEATURE EXTRACTOR")
        print("="*80)

        vgg = models.vgg19(pretrained=True).features.to(device).eval()

        # Map layer names
        self.layers = {
            'conv1_1': 0,
            'conv1_2': 2,
            'conv2_1': 5,
            'conv2_2': 7,
            'conv3_1': 10,
            'conv3_2': 12,
            'conv3_3': 14,
            'conv3_4': 16,
            'conv4_1': 19,
            'conv4_2': 21,
            'conv4_3': 23,
            'conv4_4': 25,
            'conv5_1': 28,
            'conv5_2': 30,
            'conv5_3': 32,
            'conv5_4': 34,
        }

        # Build sequential models up to each layer
        self.models = {}
        for name, idx in self.layers.items():
            self.models[name] = nn.Sequential(*list(vgg.children())[:idx+1])

        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False

        print("VGG19 loaded and frozen")
        print(f"Style layers: {STYLE_LAYERS}")
        print(f"Content layers: {CONTENT_LAYERS}")

    def forward(self, x, layers):
        """Extract features from specified layers"""
        features = {}
        for layer in layers:
            if layer in self.models:
                features[layer] = self.models[layer](x)
        return features

def gram_matrix(features):
    """
    Compute Gram matrix for style representation

    Gram matrix captures correlations between feature maps
    G_ij = sum_k F_ik * F_jk
    """
    b, c, h, w = features.shape
    features = features.view(b, c, h * w)

    # Compute Gram matrix
    gram = torch.bmm(features, features.transpose(1, 2))

    # Normalize by number of elements
    gram = gram / (c * h * w)

    return gram

def content_loss(target_features, content_features):
    """Compute content loss (MSE between features)"""
    loss = 0
    for layer in CONTENT_LAYERS:
        loss += nn.functional.mse_loss(
            target_features[layer],
            content_features[layer]
        )
    return loss

def style_loss(target_features, style_features):
    """Compute style loss (MSE between Gram matrices)"""
    loss = 0
    for layer in STYLE_LAYERS:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])

        loss += nn.functional.mse_loss(target_gram, style_gram)

    return loss / len(STYLE_LAYERS)

def total_variation_loss(img):
    """
    Total variation loss for smoothness

    Reduces noise by penalizing differences between adjacent pixels
    """
    b, c, h, w = img.shape

    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()

    return (tv_h + tv_w) / (b * c * h * w)

def load_image(image_path, image_size=512, device='cpu'):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    return image.to(device)

def save_image(tensor, output_path):
    """Convert tensor to image and save"""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)

    image = tensor.squeeze(0) * std + mean
    image = torch.clamp(image, 0, 1)

    # Convert to PIL
    image = transforms.ToPILImage()(image.cpu())
    image.save(output_path)

def create_sample_images():
    """Create sample content and style images"""
    print("\n" + "="*80)
    print("CREATING SAMPLE IMAGES")
    print("="*80)

    sample_dir = Path('data/style_transfer_samples')
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Content image (simple scene)
    content_img = Image.new('RGB', (512, 512))
    pixels = content_img.load()

    for y in range(512):
        for x in range(512):
            # Create a simple landscape
            if y < 200:  # Sky
                pixels[x, y] = (100, 150, 255)
            elif y < 300:  # Mountains
                offset = abs(x - 256) / 5
                if y < 300 - offset:
                    pixels[x, y] = (100, 100, 100)
                else:
                    pixels[x, y] = (50, 200, 50)
            else:  # Ground
                pixels[x, y] = (50, 200, 50)

    content_path = sample_dir / 'content.jpg'
    content_img.save(content_path)

    # Style image (abstract pattern)
    style_img = Image.new('RGB', (512, 512))
    pixels = style_img.load()

    for y in range(512):
        for x in range(512):
            # Create wavy pattern
            r = int(128 + 127 * np.sin(x / 20))
            g = int(128 + 127 * np.sin(y / 20))
            b = int(128 + 127 * np.sin((x + y) / 30))
            pixels[x, y] = (r, g, b)

    style_path = sample_dir / 'style.jpg'
    style_img.save(style_path)

    print(f"Created content image: {content_path}")
    print(f"Created style image: {style_path}")

    return str(content_path), str(style_path)

def neural_style_transfer(content_path, style_path, device='cpu'):
    """
    Perform neural style transfer

    Optimizes a target image to match content of one image
    and style of another
    """
    print("\n" + "="*80)
    print("NEURAL STYLE TRANSFER")
    print("="*80)

    # Load images
    print("\nLoading images...")
    content_img = load_image(content_path, CONFIG['image_size'], device)
    style_img = load_image(style_path, CONFIG['image_size'], device)

    print(f"Content image: {content_img.shape}")
    print(f"Style image: {style_img.shape}")

    # Initialize target image (start from content)
    target_img = content_img.clone().requires_grad_(True)

    # Load VGG feature extractor
    feature_extractor = VGGFeatureExtractor(device)

    # Extract content and style features
    print("\nExtracting features...")
    with torch.no_grad():
        content_features = feature_extractor(content_img, CONTENT_LAYERS)
        style_features = feature_extractor(style_img, STYLE_LAYERS)

    # Optimizer
    optimizer = optim.LBFGS([target_img])

    # Training loop
    print("\nOptimizing...")
    print(f"Steps: {CONFIG['num_steps']}")
    print(f"Content weight: {CONFIG['content_weight']}")
    print(f"Style weight: {CONFIG['style_weight']}")
    print("=" * 80)

    history = {'total': [], 'content': [], 'style': [], 'tv': []}

    step = [0]  # Use list to modify in closure

    progress_bar = tqdm(range(CONFIG['num_steps']))

    def closure():
        optimizer.zero_grad()

        # Extract features from target
        target_features = feature_extractor(
            target_img,
            CONTENT_LAYERS + STYLE_LAYERS
        )

        # Compute losses
        c_loss = content_loss(target_features, content_features)
        s_loss = style_loss(target_features, style_features)
        tv_loss = total_variation_loss(target_img)

        # Total loss
        total_loss = (
            CONFIG['content_weight'] * c_loss +
            CONFIG['style_weight'] * s_loss +
            CONFIG['tv_weight'] * tv_loss
        )

        total_loss.backward()

        # Record losses
        history['total'].append(total_loss.item())
        history['content'].append(c_loss.item())
        history['style'].append(s_loss.item())
        history['tv'].append(tv_loss.item())

        if step[0] % 50 == 0:
            progress_bar.set_postfix({
                'total': total_loss.item(),
                'content': c_loss.item(),
                'style': s_loss.item()
            })

        step[0] += 1
        progress_bar.update(1)

        return total_loss

    # Optimize
    for _ in range(CONFIG['num_steps'] // 10):
        optimizer.step(closure)

    progress_bar.close()

    return target_img, history

def visualize_results(content_path, style_path, result_img, history):
    """Visualize style transfer results"""
    print("\n" + "="*80)
    print("VISUALIZING RESULTS")
    print("="*80)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save result
    result_path = output_dir / 'stylized.jpg'
    save_image(result_img, result_path)
    print(f"\nSaved stylized image to {result_path}")

    # Load images for display
    content_img = Image.open(content_path)
    style_img = Image.open(style_path)
    result_img_pil = Image.open(result_path)

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(content_img)
    axes[0].set_title('Content Image')
    axes[0].axis('off')

    axes[1].imshow(style_img)
    axes[1].set_title('Style Image')
    axes[1].axis('off')

    axes[2].imshow(result_img_pil)
    axes[2].set_title('Stylized Result')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_dir / 'comparison.png'}")

    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['content'])
    axes[0, 1].set_title('Content Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['style'])
    axes[1, 0].set_title('Style Loss')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['tv'])
    axes[1, 1].set_title('Total Variation Loss')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved loss curves to {output_dir / 'loss_curves.png'}")

def main():
    print("="*80)
    print("NEURAL STYLE TRANSFER")
    print("="*80)

    print(f"\nDevice: {CONFIG['device']}")

    # Create sample images
    content_path, style_path = create_sample_images()

    # Perform style transfer
    result_img, history = neural_style_transfer(
        content_path,
        style_path,
        CONFIG['device']
    )

    # Visualize
    visualize_results(content_path, style_path, result_img, history)

    print("\n" + "="*80)
    print("STYLE TRANSFER COMPLETED")
    print("="*80)

    print("\nKey Concepts:")
    print("✓ Content loss: Match high-level features")
    print("✓ Style loss: Match Gram matrices (feature correlations)")
    print("✓ VGG features: Pre-trained representations")
    print("✓ Optimization: Iterate on image pixels")

    print("\nGram Matrix:")
    print("- Captures style (textures, colors, patterns)")
    print("- Correlations between feature maps")
    print("- Ignores spatial structure")

    print("\nLoss Weights:")
    print("- Higher style weight → more stylized")
    print("- Higher content weight → more faithful to content")
    print("- TV weight → smoother result")

    print("\nApplications:")
    print("- Artistic filters and effects")
    print("- Photo enhancement")
    print("- Video stylization")
    print("- Creative tools")

    print("\nVariants:")
    print("- Fast neural style (feed-forward network)")
    print("- Arbitrary style transfer (AdaIN)")
    print("- Video style transfer (temporal consistency)")

if __name__ == '__main__':
    main()

import torch
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np

# Load the SDXL VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

device = "cuda" if torch.cuda.is_available() else "cpu"
vae.to(device)

# Create random latent noise with the correct shape
# SDXL VAE typically expects latents of shape:
# (batch, latent_channels, height/8, width/8)
# latent_channels for SDXL is usually 4
random_latents = torch.randn(1, 4, 128, 128).to(device) /  vae.config.scaling_factor  # scale by VAE latent scaling factor

# Optional: scale the latents if needed
# random_latents = random_latents * 0.18215

with torch.no_grad():
    decoded = vae.decode(random_latents).sample

# Convert to uint8 image
img = ((decoded.clamp(-1, 1) + 1) * 127.5).cpu().numpy().astype(np.uint8)
print(img.shape)
img = np.transpose(img, (0, 2, 3, 1))[0]  # NHWC
im = Image.fromarray(img)
im.save("random_decoded.png")
im.show()

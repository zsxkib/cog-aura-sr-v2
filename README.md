# AuraSR v2: Advanced GAN-based Super-Resolution 🖼️
[![Replicate](https://replicate.com/zsxkib/aura-sr-v2/badge)](https://replicate.com/zsxkib/aura-sr-v2) 
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-AuraSR--v2-blue)](https://huggingface.co/fal/AuraSR-v2)

AuraSR v2 is a second-generation GAN-based Super-Resolution model designed for real-world applications. It's an improved version based on the GigaGAN paper, specifically optimized for upscaling generated images.

![See AuraSR in action](https://storage.googleapis.com/falserverless/gallery/aurasr-animated.webp)

Certainly! I'll update the Quick Start section to focus on cloning the repository, installing Cog, and running a prediction. Here's the revised version:

## Quick Start 🚀

1. Clone the repository:
```bash
git clone git@github.com:zsxkib/cog-aura-sr-v2.git
cd cog-aura-sr-v2
```

2. Install Cog:
```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

3. Run a prediction:
```bash
cog predict -i 'image=@example_input.png'
```
or if it's a url to an image you can do
```bash
cog predict -i image="https://example.com/your_image.png"
```


## What It Does 🎨

- Upscales images using advanced GAN-based super-resolution techniques
- Optimized for generated images and high-quality photos
- Supports 4x upscaling with overlapped processing for improved results

## Model Details 📊

- Model size: 618M parameters
- Tensor type: F32
- License: Apache 2.0
  - But this code is MIT (feel free to do whatever you want with it)

## How to Use It 🛠️

1. Install the `aura-sr` package
2. Load the model using `AuraSR.from_pretrained("fal/AuraSR-v2")`
3. Use the `upscale_4x_overlapped` method to upscale your images

For detailed usage examples, check the [Hugging Face model page](https://huggingface.co/fal/AuraSR-v2).

## Thank You 🙌
- The amazing people at [Fal.ai](https://fal.ai/) for creating this model
- [GigaGAN paper](https://mingukkang.github.io/GigaGAN/) for the foundational work on image-conditioned upscaling
- [lucidrains/gigagan-pytorch](https://github.com/lucidrains/gigagan-pytorch) for the unofficial PyTorch implementation

## Community and Support 🤝

- Check out the [Hugging Face community tab](https://huggingface.co/fal/AuraSR-v2/discussions) for discussions and support
- Explore [Spaces using AuraSR-v2](https://huggingface.co/fal/AuraSR-v2#spaces-usingfalaurasr-v2-4) for implementation examples

## License 📄

AuraSR v2 is released under the Apache 2.0 license.

## Previous Version 🔙

Looking for AuraSR v1? Check out the [original AuraSR repository](https://github.com/zsxkib/cog-aura-sr) for the first version of this GAN-based Super-Resolution model.
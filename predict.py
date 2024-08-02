import os
import time
import torch
import mimetypes
import subprocess
from PIL import Image
from aura_sr import AuraSR
from cog import BasePredictor, Input, Path

mimetypes.add_type("image/webp", ".webp")

MODEL_CACHE = "checkpoints"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE
BASE_URL = f"https://weights.replicate.delivery/default/aura-sr/{MODEL_CACHE}/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.get_default_dtype()


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        model_files = [
            "models--fal--AuraSR-v2.tar",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        torch.set_default_dtype(DTYPE)
        torch.set_default_device(DEVICE)

        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(
            *args, **kwargs, map_location=DEVICE
        )

        self.model = AuraSR.from_pretrained("fal/AuraSR-v2")
        torch.load = original_load

    def predict(
        self,
        image: Path = Input(description="Input image to upscale"),
        max_batch_size: int = Input(
            description="Maximum number of tiles to process in a single batch. Higher values may increase speed but require more GPU memory.",
            default=8,
            ge=1,
            le=64,
        ),
        output_format: str = Input(
            description="The image file format of the generated output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="The image compression quality (for lossy formats like JPEG and WebP). 100 = best quality, 0 = lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        input_image = Image.open(image)
        output_image = self.model.upscale_4x(input_image, max_batch_size=max_batch_size)

        extension = output_format.lower()
        extension = "jpeg" if extension == "jpg" else extension
        output_path = Path(f"output.{extension}")

        save_params = {"format": extension.upper()}
        if output_format != "png":
            save_params["quality"] = output_quality
            save_params["optimize"] = True

        output_image.save(output_path, **save_params)
        return output_path

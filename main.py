from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
import warnings
from PIL import Image
import torch
import numpy as np
from transformers import GPT2TokenizerFast, ViTFeatureExtractor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
warnings.filterwarnings('ignore')
model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer= GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
def show_n_generate(image_path, greedy=True, model=None):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize the image to the expected size
    image_tensor = image_processor(images=image, return_tensors="pt").pixel_values
    plt.imshow(np.asarray(image))
    plt.show()

    if greedy:
        generated_ids = model.generate(image_tensor, max_length=30)
    else:
        generated_ids = model.generate(
            image_tensor,
            do_sample=True,
            max_length=30,
            top_k=5
        )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

image_path = "image/image3.png"
show_n_generate(image_path, greedy=False, model=model_raw)



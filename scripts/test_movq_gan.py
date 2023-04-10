from argparse import ArgumentParser

import torch
import numpy as np
from PIL import Image
from muse.modeling_maskgit_movq import MOVQ

def generate_and_log(args):
    image_encoder = MOVQ()
    image_encoder.load_state_dict(torch.load(args.movq_path))
    image_encoder = image_encoder.eval().to(args.device)

    image = Image.open(args.input_img).resize((256, 256))
    image = np.array(image.convert("RGB"))
    image = image.astype(np.float32) / 127.5 - 1
    image = np.transpose(image, [2, 0, 1])
    image = torch.from_numpy(image)
    _, ids = image_encoder.encode(image[None].to(args.device))
    recon = (image_encoder.decode_code(ids).cpu().detach().numpy()+1) * 127.5
    recon = np.transpose(recon[0], [1, 2, 0]).astype(np.uint8)
    recon = Image.fromarray(recon)
    recon.save(args.output_img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_img", type=str, required=True, help="path to input image")
    parser.add_argument("--output_img", type=str, default="output.jpg", help="path to output image")
    parser.add_argument("--movq_path", type=str, required=True, help="Path to movq model")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    generate_and_log(args)
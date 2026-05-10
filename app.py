import streamlit as st
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
import yaml

from src.utils.visualize import plot_translation_results
from src.utils.metrics import MetricsCalculator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_calc = MetricsCalculator(device=device)

st.set_page_config(page_title="CrossModalMedNet - SynthRAD 2023", layout="wide")
st.title("CrossModalMedNet: CT to MRI Translation (8 Research Models)")

base_dir = os.path.abspath(os.path.dirname(__file__))

def load_config(config_name):
    with open(os.path.join(base_dir, "configs", config_name), 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_model_instance(architecture, region):
    if architecture == "CycleGAN 2D":
        config = load_config(f"cyclegan2d_{region.lower()}.yaml")
        model_path = os.path.join(base_dir, "models", "cyclegan")
        if model_path not in sys.path: sys.path.insert(0, model_path)
        from models import build_cyclegan_2d_friendly
        model = build_cyclegan_2d_friendly(use_attention=False, use_multiscale=True)
        ckpt_path = os.path.join(base_dir, "checkpoints", f"cyclegan_{region.lower()}", "latest.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
            model.eval().to(device)
            return model, ckpt_path
    
    elif architecture == "Pix2Pix 2D":
        config = load_config(f"pix2pix2d_{region.lower()}.yaml")
        model_path = os.path.join(base_dir, "models", "pix2pix")
        if model_path not in sys.path: sys.path.insert(0, model_path)
        from models import build_pix2pix
        model = build_pix2pix(input_nc=1, output_nc=1)
        ckpt_path = os.path.join(base_dir, "checkpoints", f"pix2pix_{region.lower()}", "latest.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
            model.eval().to(device)
            return model, ckpt_path

    elif architecture == "Diffusion 2.5D (Paired)":
        config = load_config(f"paired_diffusion2.5d_{region.lower()}.yaml")
        model_path = os.path.join(base_dir, "models", "diffusion")
        if model_path not in sys.path: sys.path.insert(0, model_path)
        from configs.config import get_settings
        from model import initialize_unet, initialize_vae, load_lora_checkpoint, make_1step_sched, forward_with_networks, VAEEncode, VAEDecode
        from transformers import AutoTokenizer, CLIPTextModel
        settings = get_settings()
        unet = initialize_unet(settings.base_model, settings.lora_rank_unet, add_lora=False)
        vae = initialize_vae(settings.base_model, settings.lora_rank_vae, add_lora=False)
        ckpt_path = os.path.join(base_dir, "checkpoints", f"diffusion_{region.lower()}", "latest.pt")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location=device)
            load_lora_checkpoint(unet, sd, "unet")
            load_lora_checkpoint(vae, sd, "vae")
            unet.eval().to(device); vae.eval().to(device)
            scheduler = make_1step_sched(settings.base_model, device)
            tokenizer = AutoTokenizer.from_pretrained(settings.base_model, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(settings.base_model, subfolder="text_encoder").to(device)
            return {"unet": unet, "vae_enc": VAEEncode(vae), "vae_dec": VAEDecode(vae), "scheduler": scheduler, "tokenizer": tokenizer, "text_encoder": text_encoder, "settings": settings, "forward_with_networks": forward_with_networks}, ckpt_path

    elif architecture == "Diffusion 2.5D (Unpaired)":
        config = load_config(f"unpaired_diffusion2.5d_{region.lower()}.yaml")
        model_path = os.path.join(base_dir, "models", "diffusion_unpaired")
        if model_path not in sys.path: sys.path.insert(0, model_path)
        from config import get_settings
        from model import initialize_unet, initialize_vae, load_lora_checkpoint, make_1step_sched, forward_with_networks, VAEEncode, VAEDecode
        from transformers import AutoTokenizer, CLIPTextModel
        settings = get_settings()
        unet = initialize_unet(settings.base_model, settings.lora_rank_unet, add_lora=True)
        vae = initialize_vae(settings.base_model, settings.lora_rank_vae, add_lora=True)
        ckpt_path = os.path.join(base_dir, "checkpoints", f"diffusion_{region.lower()}_unpaired", "latest.pt")
        if os.path.exists(ckpt_path):
            vae_b2a, vae_enc, vae_dec, _ = load_lora_checkpoint(ckpt_path, unet, vae)
            unet.eval().to(device); vae.eval().to(device); vae_b2a.eval().to(device)
            scheduler = make_1step_sched(settings.base_model, device)
            tokenizer = AutoTokenizer.from_pretrained(settings.base_model, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(settings.base_model, subfolder="text_encoder").to(device)
            return {"unet": unet, "vae_enc": vae_enc, "vae_dec": vae_dec, "scheduler": scheduler, "tokenizer": tokenizer, "text_encoder": text_encoder, "settings": settings, "forward_with_networks": forward_with_networks}, ckpt_path
            
    return None, None

# Sidebar
st.sidebar.title("Model Selection")
arch = st.sidebar.selectbox("Architecture", ["CycleGAN 2D", "Pix2Pix 2D", "Diffusion 2.5D (Paired)", "Diffusion 2.5D (Unpaired)"])
reg = st.sidebar.selectbox("Region", ["Brain", "Pelvis"])

with st.spinner(f"Loading {arch} ({reg})..."):
    model_info, ckpt_path = load_model_instance(arch, reg)

if ckpt_path:
    st.sidebar.success(f"Loaded: {os.path.basename(ckpt_path)}")
else:
    st.sidebar.error("Checkpoint not found!")

# Inference
st.write("### CT to MRI Inference")
uploaded_file = st.file_uploader("Upload CT Slice (.pt, .npy, .png, .jpg)", type=["pt", "npy", "png", "jpg"])

if uploaded_file and model_info:
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == "pt": img_tensor = torch.load(uploaded_file, map_location="cpu")
        elif ext == "npy": img_tensor = torch.from_numpy(np.load(uploaded_file)).float()
        else:
            img_np = np.array(Image.open(uploaded_file).convert("L")).astype(np.float32)
            img_tensor = torch.from_numpy(img_np) / 127.5 - 1.0
            
        if img_tensor.dim() == 2: img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        elif img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
        
        if img_tensor.shape[-2:] != (256, 256):
            img_tensor = F.interpolate(img_tensor, size=(256, 256), mode='bilinear')
            
        # Normalization (simple fallback)
        img_input = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8) * 2 - 1
        img_input = img_input.to(device)
        
        with torch.no_grad():
            if "CycleGAN" in arch or "Pix2Pix" in arch:
                output = model_info.G_CT2MRI(img_input)
            elif "Diffusion 2.5D (Paired)" in arch:
                diff = model_info
                prompt = diff["settings"].prompt_target
                tokens = diff["tokenizer"](prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids.to(device)
                emb = diff["text_encoder"](tokens)[0]
                timesteps = torch.tensor([1], dtype=torch.long, device=device)
                # repeat 1 channel to 3 for SD
                output = diff["forward_with_networks"](img_input.repeat(1,3,1,1), diff["vae_enc"], diff["unet"], diff["vae_dec"], diff["scheduler"], timesteps, emb)[:, 0:1]
            else: # Unpaired Diffusion
                diff = model_info
                prompt = diff["settings"].prompt_target
                tokens = diff["tokenizer"](prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids.to(device)
                emb = diff["text_encoder"](tokens)[0]
                timesteps = torch.tensor([1], dtype=torch.long, device=device)
                output = diff["forward_with_networks"](img_input.repeat(1,3,1,1), diff["vae_enc"], diff["unet"], diff["vae_dec"], diff["scheduler"], timesteps, emb, direction="a2b")[:, 0:1]

        col1, col2 = st.columns(2)
        with col1: st.image((img_input[0,0].cpu().numpy() + 1)/2, caption="Input CT", use_container_width=True)
        with col2: st.image((output[0,0].cpu().numpy() + 1)/2, caption="Generated MRI", use_container_width=True)
            
    except Exception as e:
        st.error(f"Inference Error: {e}")

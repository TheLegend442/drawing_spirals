import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- import the model the way the authors indicate on the HF model card ---
# HF card says to use model_specs + PhysicsTransformer via a get_model(...)
# We mirror that here. If these imports ever change upstream, run `pip show gphyt`
# and peek into gphyt/model/* to adjust.
from gphyt.model import model_specs  # provides GPT_S/M/L/XL spec classes
from gphyt.model.transformer.model import PhysicsTransformer  # the core module

# ---------------------- CONFIG (tune to your checkpoint) ----------------------
MODEL_SIZE = "GPT_M"          # <<< IMPORTANT >>> one of: GPT_S, GPT_M, GPT_L, GPT_XL
IMG_SIZE   = (1, 128, 128)   # (T, H, W) -> T=1 for static spirals
PATCH_SIZE = (1, 8, 8)       # (t_patch, h_patch, w_patch)

IN_CHANNELS = 5              # keep this matching your checkpoint
USE_DERIVS = True             # whether model expects derivative channels
POS_ENC    = "absolute"          # typical; repo supports others via config

TOKENIZER_MODE     = "patch"    # defaults used in repo
DETOKENIZER_MODE   = "patch"
TOKENIZER_OVERLAP  = 0
DETOKENIZER_OVERLAP= 0
ATT_MODE           = "full"    # full attention (others exist in repo)
INTEGRATOR         = "Euler"   # time integrator name (not used in pure AE pass)
DROPOUT            = 0.0
STOCH_DEPTH        = 0.0

T,H,W = IMG_SIZE
pt,ph,pw = PATCH_SIZE
assert (T%pt==0) and (H%ph==0) and (W%pw==0), "img_size must be divisible by patch_size"

# Path to your downloaded state dict (.pth)
CKPT_PATH = Path("General-Physics-Transformer/models/gphyt-M.pth")  # <<< IMPORTANT >>> set to your file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_config: dict) -> torch.nn.Module:
    """Same pattern the authors show on the HF card."""
    transformer_config: dict = model_config["transformer"]
    tokenizer_config: dict = model_config["tokenizer"]

    if transformer_config["model_size"] == "GPT_S":
        gpt_cfg = model_specs.GPT_S()
    elif transformer_config["model_size"] == "GPT_M":
        gpt_cfg = model_specs.GPT_M()
    elif transformer_config["model_size"] == "GPT_L":
        gpt_cfg = model_specs.GPT_L()
    elif transformer_config["model_size"] == "GPT_XL":
        gpt_cfg = model_specs.GPT_XL()
    else:
        raise ValueError(f"Invalid model size: {transformer_config['model_size']}")

    return PhysicsTransformer(
        num_fields=transformer_config["input_channels"],
        hidden_dim=gpt_cfg.hidden_dim,
        mlp_dim=gpt_cfg.mlp_dim,
        num_heads=gpt_cfg.num_heads,
        num_layers=gpt_cfg.num_layers,
        att_mode=transformer_config.get("att_mode", "full"),
        integrator=transformer_config.get("integrator", "Euler"),
        pos_enc_mode=transformer_config["pos_enc_mode"],
        img_size=model_config["img_size"],
        patch_size=transformer_config["patch_size"],
        use_derivatives=transformer_config["use_derivatives"],
        tokenizer_mode=tokenizer_config["tokenizer_mode"],
        detokenizer_mode=tokenizer_config["detokenizer_mode"],
        tokenizer_overlap=tokenizer_config["tokenizer_overlap"],
        detokenizer_overlap=tokenizer_config["detokenizer_overlap"],
        tokenizer_net_channels=gpt_cfg.conv_channels,
        detokenizer_net_channels=gpt_cfg.conv_channels,
        dropout=transformer_config["dropout"],
        stochastic_depth_rate=transformer_config["stochastic_depth_rate"],
    )


def build_config():
    return {
        "img_size": IMG_SIZE,
        "transformer": {
            "model_size": MODEL_SIZE,
            "input_channels": IN_CHANNELS,
            "patch_size": PATCH_SIZE,
            "pos_enc_mode": POS_ENC,
            "use_derivatives": USE_DERIVS,
            "att_mode": ATT_MODE,
            "integrator": INTEGRATOR,
            "dropout": DROPOUT,
            "stochastic_depth_rate": STOCH_DEPTH,
        },
        "tokenizer": {
            "tokenizer_mode": TOKENIZER_MODE,
            "detokenizer_mode": DETOKENIZER_MODE,
            "tokenizer_overlap": TOKENIZER_OVERLAP,
            "detokenizer_overlap": DETOKENIZER_OVERLAP,
        },
    }


# -------------------------- Spiral data generation ---------------------------
def make_spiral(batch=16, img_size=(1,128,128), channels=5, noise=0.03, seed=0):
    """
    Returns [B, C, T, H, W]. We draw the spiral in the first time slice (t=0).
    """
    if isinstance(img_size, int):
        T, H, W = (1, img_size, img_size)
    else:
        T, H, W = img_size

    import torch
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, dtype=torch.float32),
        torch.linspace(-1, 1, W, dtype=torch.float32),
        indexing="ij",
    )
    rr = torch.sqrt(xx**2 + yy**2) + 1e-8
    theta = torch.atan2(yy, xx)

    # archimedean-ish spiral
    a, b = 0.0, 0.2 * math.pi
    target_r = a + b * (theta + 2 * math.pi * 2) / (2 * math.pi)
    dist = torch.abs(rr - (target_r / target_r.abs().max().clamp_min(1e-6)))

    spiral = torch.exp(- (dist / 0.03)**2)
    spiral = (spiral - spiral.min()) / (spiral.max() - spiral.min() + 1e-8)
    spiral = (spiral + noise * torch.randn_like(spiral, generator=g)).clamp(0.0, 1.0)

    x = torch.zeros(batch, channels, T, H, W)
    x[:, 0, 0] = spiral  # first channel, first time slice
    return x



# -------------------------- Latent encode/decode -----------------------------
def find_tokenizer_and_detokenizer(model: torch.nn.Module):
    """
    Try common attribute names from the repo; fall back to scanning children.
    """
    tok = getattr(model, "tokenizer", None)
    detok = getattr(model, "detokenizer", None)
    if tok is not None and detok is not None:
        return tok, detok

    # fallback: search by substring
    tok, detok = None, None
    for name, mod in model.named_modules():
        lname = name.lower()
        if tok is None and "token" in lname and hasattr(mod, "forward"):
            tok = mod
        if detok is None and ("detoken" in lname or "decoder" in lname) and hasattr(mod, "forward"):
            detok = mod
    if tok is None or detok is None:
        raise RuntimeError("Could not find tokenizer/detokenizer on the model. "
                           "Inspect the module tree and adjust the attribute names.")
    return tok, detok


@torch.no_grad()
def encode_decode(model, x):
    """
    Returns latent tokens and reconstruction.
    """
    model.eval()
    tok, detok = find_tokenizer_and_detokenizer(model)

    # Many tokenizers expect [B, C, H, W]; some add positional encodings internally.
    z = tok(x)                      # latent tokens
    x_rec = detok(z, out_size=x.shape[-3:]) if "out_size" in detok.forward.__code__.co_varnames else detok(z)
    return z, x_rec


def pca_latents(z):
    """
    Flatten tokens per sample and run 2D PCA for quick visualization.
    z: [B, T, D] or [B, N, D] or [B, D, H', W'] -> we flatten to [B, -1]
    """
    if z.dim() == 4:               # [B, D, H', W'] -> [B, D*H'*W']
        z_flat = z.flatten(1)
    elif z.dim() == 3:             # [B, N, D] or [B, T, D]
        z_flat = z.flatten(1)
    else:
        z_flat = z.view(z.shape[0], -1)

    pca = PCA(n_components=2, random_state=0)
    emb2 = pca.fit_transform(z_flat.cpu().numpy())
    return emb2, pca.explained_variance_ratio_


def main():
    # ---------------- model ----------------
    cfg = build_config()
    model = get_model(cfg).to(DEVICE)

    # --- load user checkpoint (state dict) ---
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # Accept either plain state_dict or dict with 'model' / 'state_dict' keys
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model", "model_state_dict")):
        state = ckpt.get("state_dict", ckpt.get("model", ckpt.get("model_state_dict")))
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights from {CKPT_PATH}")
    if missing or unexpected:
        print("Missing keys:", missing[:10], ("..." if len(missing) > 10 else ""))
        print("Unexpected keys:", unexpected[:10], ("..." if len(unexpected) > 10 else ""))

    # ---------------- data ----------------
    x = make_spiral(batch=32, img_size=IMG_SIZE, channels=IN_CHANNELS).to(DEVICE)

    # ---------------- encode/decode ----------------
    z, x_rec = encode_decode(model, x)

    # ---------------- quick sanity plots ----------------
    b0 = 0
    src = x[b0, 0].detach().cpu().numpy()
    rec = x_rec[b0, 0].detach().cpu().numpy()
    vmin = min(src.min(), rec.min()); vmax = max(src.max(), rec.max())

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.title("Spiral (ch0)"); plt.imshow(src, vmin=vmin, vmax=vmax); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Reconstruction (ch0)"); plt.imshow(rec, vmin=vmin, vmax=vmax); plt.axis("off")
    plt.subplot(1,3,3); plt.title("|Error|"); plt.imshow(np.abs(src-rec)); plt.axis("off")
    plt.tight_layout(); plt.show()

    # ---------------- latent analysis ----------------
    emb2, evr = pca_latents(z)
    print(f"PCA variance explained: {evr}")

    plt.figure()
    plt.scatter(emb2[:,0], emb2[:,1], s=18, alpha=0.8)
    plt.title("Latent tokens (per-sample) — PCA2")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.show()


if __name__ == "__main__":
    main()
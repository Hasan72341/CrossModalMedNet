import os
import subprocess

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
TECTONIC_BIN = os.environ.get("TECTONIC_BIN", "tectonic")

# -------------------- Style Constants (from mid_project_report.tex) --------------------
PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.62in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage{subcaption}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[most]{tcolorbox}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc,shapes.geometric}

\definecolor{brand}{HTML}{0D3B66}
\definecolor{accent}{HTML}{2A9D8F}
\definecolor{softbg}{HTML}{F4F8FB}
\definecolor{textdark}{HTML}{1E1E1E}

\hypersetup{
  colorlinks=true,
  linkcolor=brand,
  urlcolor=accent
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\setlist[itemize]{leftmargin=1.2em,itemsep=1pt,topsep=2pt}

\titleformat{\section}{\color{brand}\large\bfseries}{\thesection.}{0.4em}{}
\titleformat{\subsection}{\color{textdark}\bfseries}{\thesubsection}{0.4em}{}
\titlespacing*{\section}{0pt}{8pt}{4pt}
\titlespacing*{\subsection}{0pt}{6pt}{3pt}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small CS671 Technical Model Report}
\fancyhead[R]{\small %(model_name)s}
\fancyfoot[C]{\small \thepage}
\setlength{\headheight}{14pt}

\begin{document}

% -------------------- Header Block --------------------
\begin{tcolorbox}[
  enhanced,
  colback=softbg,
  colframe=brand,
  boxrule=0.9pt,
  arc=2mm,
  left=5pt,right=5pt,top=5pt,bottom=5pt
]
\centering
{\Large\bfseries\color{brand} CS671 -- Deep Learning and Its Applications}\\[2pt]
{\large\bfseries Technical Documentation Report}\\[6pt]
{\normalsize\bfseries Model: %(model_title)s}\\[4pt]
\small
Project Group: 5 \quad Region: %(region)s \quad Modality: %(modality)s
\end{tcolorbox}

"""

POSTAMBLE = r"""
\end{document}
"""

# -------------------- Model Specific Templates --------------------

def get_cyclegan_content(region):
    return r"""
\section{ARCHITECTURAL OVERVIEW}
CycleGAN enables unpaired image-to-image translation by learning a mapping $G: X \to Y$ and $F: Y \to X$ such that the dynamics of the domains are preserved through cycle consistency. In this specific implementation for %(region)s, the model translates between CT and MRI slices without requiring direct spatial alignment.

\subsection{Generator Architecture (ResNet-9)}
The generator uses a deep residual architecture to prevent vanishing gradients and preserve high-frequency anatomical details.

\begin{figure}[H]
\centering
\resizebox{0.9\textwidth}{!}{
\begin{tikzpicture}[
    node distance=10mm,
    block/.style={draw=brand, thick, fill=softbg, minimum width=2.5cm, minimum height=1cm, align=center, rounded corners},
    tensor/.style={font=\scriptsize\ttfamily, color=brand!70},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[block] (in) {Input Slice\\(1, 256, 256)};
    \node[block, right=of in] (enc) {Encoder\\7x7 Conv, Stride 1\\64 filters};
    \node[block, right=of enc] (down1) {Downsampling\\3x3 Conv, Stride 2\\128 filters};
    \node[block, right=of down1] (down2) {Downsampling\\3x3 Conv, Stride 2\\256 filters};
    
    \node[block, below=1.5cm of down2] (res1) {9x Residual Blocks\\3x3 Conv, BN, ReLU\\256 filters};
    
    \node[block, left=of res1] (up1) {Upsampling\\3x3 Deconv, Stride 2\\128 filters};
    \node[block, left=of up1] (up2) {Upsampling\\3x3 Deconv, Stride 2\\64 filters};
    \node[block, left=of up2] (out) {Output Layer\\7x7 Conv, Tanh\\(1, 256, 256)};

    \draw[arr] (in) -- (enc);
    \draw[arr] (enc) -- (down1);
    \draw[arr] (down1) -- (down2);
    \draw[arr] (down2) -- (res1);
    \draw[arr] (res1) -- (up1);
    \draw[arr] (up1) -- (up2);
    \draw[arr] (up2) -- (out);
    
    \node[tensor, above=1mm of in] {Raw Tensor};
    \node[tensor, above=1mm of down1] {128x128};
    \node[tensor, above=1mm of down2] {64x64};
    \node[tensor, below=1mm of res1] {Bottleneck};
    \node[tensor, below=1mm of out] {Synthetic MRI};
\end{tikzpicture}
}
\caption{CycleGAN Generator Architecture with 9 Residual Blocks.}
\end{figure}

\textbf{Encoder \& Downsampling:} The input $256 \times 256$ slice is first processed by a $7 \times 7$ reflective padded convolution to extract initial features, followed by two stride-2 convolutions that reduce spatial dimensions to $64 \times 64$ while increasing the feature depth to 256. This compression forces the network to learn compact latent representations of the anatomy.

\textbf{Residual Bottleneck:} The core of the generator consists of 9 residual blocks. Each block employs a "skip connection" where the input is added to the output of two $3 \times 3$ convolutions. This allows the gradient to flow directly through the network, enabling the training of deeper architectures that can represent complex non-linear mappings between CT and MRI modalities.

\textbf{Decoder \& Reconstruction:} Two transpose convolutions perform learned upsampling back to the original resolution. The final $7 \times 7$ convolution with a Tanh activation produces the synthetic image, ensuring the output pixel intensities are normalized.

\section{TRAINING WORKFLOW}
The training objective is to optimize the generators $G, F$ and discriminators $D_X, D_Y$ through a combination of adversarial and structural losses.

\begin{figure}[H]
\centering
\resizebox{0.8\textwidth}{!}{
\begin{tikzpicture}[
    node distance=12mm,
    block/.style={draw=brand, thick, fill=softbg, minimum width=2.2cm, minimum height=0.8cm, align=center, rounded corners},
    loss/.style={draw=accent, thick, fill=accent!10, minimum width=2cm, minimum height=0.7cm, align=center},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[block] (realX) {Real CT ($x$)};
    \node[block, right=of realX] (G) {Gen $G$};
    \node[block, right=of G] (fakeY) {Fake MRI ($\hat{y}$)};
    \node[block, right=of fakeY] (F) {Gen $F$};
    \node[block, right=of F] (recX) {Rec CT ($\tilde{x}$)};
    
    \node[loss, below=1cm of fakeY] (adv) {Adversarial Loss\\($D_Y$ checks $\hat{y}$)};
    \node[loss, below=1cm of recX] (cyc) {Cycle Loss\\$L_1(x, \tilde{x})$};
    
    \draw[arr] (realX) -- (G);
    \draw[arr] (G) -- (fakeY);
    \draw[arr] (fakeY) -- (F);
    \draw[arr] (F) -- (recX);
    \draw[arr] (fakeY) -- (adv);
    \draw[arr] (recX) -- (cyc);
    \draw[arr] (realX) .. controls +(0,-2) and +(-1,0) .. (cyc);
\end{tikzpicture}
}
\caption{CycleGAN Training Dynamics: Adversarial and Cycle-Consistency loops.}
\end{figure}

\textbf{Adversarial Loss:} The PatchGAN discriminators are trained to distinguish between real and synthetic slices. This encourages the generators to produce visually realistic textures that match the target modality's distribution.

\textbf{Cycle Consistency:} To prevent mode collapse and ensure anatomical preservation, the "cycle" constraint $F(G(x)) \approx x$ is enforced using an $L_1$ loss. This ensures that the mapping is bijection-like, preventing the model from hallucinating structures that cannot be reversed.

\section{INFERENCE PIPELINE}
Inference involves the sequential processing of 3D volumes as independent 2D slices.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=8mm,
    block/.style={draw=brand, thick, fill=softbg, minimum width=2.5cm, minimum height=0.8cm, align=center, rounded corners},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[block] (v) {3D CT Volume};
    \node[block, right=of v] (s) {Slice Sampling\\(2D Projection)};
    \node[block, right=of s] (n) {Norm\\$[-1, 1]$};
    \node[block, right=of n] (m) {Generator $G$};
    \node[block, below=1cm of m] (o) {Synthetic MRI\\Slices};
    \node[block, left=of o] (r) {Rescale \&\\Reconstruct};
    \node[block, left=of r] (f) {Final 3D NIfTI};

    \draw[arr] (v) -- (s);
    \draw[arr] (s) -- (n);
    \draw[arr] (n) -- (m);
    \draw[arr] (m) -- (o);
    \draw[arr] (o) -- (r);
    \draw[arr] (r) -- (f);
\end{tikzpicture}
\caption{Inference workflow for volume-to-volume translation.}
\end{figure}

The inference process begins with volume ingestion and slice extraction. Each slice is normalized and passed through the trained generator. The resulting synthetic slices are then re-stacked and rescaled to the original intensity range, producing a complete 3D MRI volume in NIfTI format.
""" % {"region": region, "MODEL": "CycleGAN"}

def get_pix2pix_content(region):
    return r"""
\section{ARCHITECTURAL OVERVIEW}
Pix2Pix uses a conditional GAN (cGAN) framework to learn the mapping from input CT to output MRI. Unlike standard GANs, the generator is conditioned on the input image, allowing for highly specific and aligned translation.

\subsection{Generator Architecture (U-Net 256)}
The model utilizes a U-Net architecture with skip connections to preserve fine-grained spatial information.

\begin{figure}[H]
\centering
\resizebox{0.9\textwidth}{!}{
\begin{tikzpicture}[
    node distance=10mm,
    block/.style={draw=brand, thick, fill=softbg, minimum width=2.2cm, minimum height=0.8cm, align=center, rounded corners},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[block] (in) {Input CT\\(1, 256, 256)};
    \node[block, right=of in] (e1) {Encoder 1\\64 filters};
    \node[block, right=of e1] (e2) {Encoder 2\\128 filters};
    \node[block, right=of e2] (e3) {Encoder 3\\256 filters};
    \node[block, below=1.5cm of e3] (b) {Bottleneck\\512 filters};
    \node[block, left=of b] (d3) {Decoder 3\\256 filters};
    \node[block, left=of d3] (d2) {Decoder 2\\128 filters};
    \node[block, left=of d2] (d1) {Decoder 1\\64 filters};
    \node[block, left=of d1] (out) {Output MRI\\(1, 256, 256)};

    \draw[arr] (in) -- (e1);
    \draw[arr] (e1) -- (e2);
    \draw[arr] (e2) -- (e3);
    \draw[arr] (e3) -- (b);
    \draw[arr] (b) -- (d3);
    \draw[arr] (d3) -- (d2);
    \draw[arr] (d2) -- (d1);
    \draw[arr] (d1) -- (out);
    
    \draw[arr, dashed, brand!50] (e1) .. controls +(0,2) and +(0,2) .. (d1);
    \draw[arr, dashed, brand!50] (e2) .. controls +(0,1) and +(0,1) .. (d2);
    \draw[arr, dashed, brand!50] (e3) .. controls +(0,0.5) and +(0,0.5) .. (d3);
    
    \node[font=\scriptsize, color=brand!70] at (0, 1.8) {Skip Connections};
\end{tikzpicture}
}
\caption{U-Net Generator Architecture with encoder-decoder skip connections.}
\end{figure}

\textbf{U-Net Encoder-Decoder:} The encoder progressively downsamples the input through 8 layers of convolutions, capturing hierarchical features. The decoder then mirrors this process to reconstruct the target MRI.

\textbf{Skip Connections:} Crucially, activations from each encoder layer are concatenated with the corresponding decoder layer. This allows the network to bypass the bottleneck for high-frequency structural information, ensuring that thin anatomical boundaries in the %(region)s are perfectly preserved.

\section{TRAINING WORKFLOW}
Training is performed using paired CT-MRI slices with a combined loss function.

\begin{figure}[H]
\centering
\resizebox{0.7\textwidth}{!}{
\begin{tikzpicture}[
    node distance=10mm,
    block/.style={draw=brand, thick, fill=softbg, minimum width=2cm, minimum height=0.7cm, align=center, rounded corners},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[block] (ct) {Input CT};
    \node[block, right=of ct] (G) {Generator};
    \node[block, right=of G] (fake) {Fake MRI};
    \node[block, below=of fake] (real) {Real MRI};
    \node[block, below=of G] (D) {Discriminator};
    \node[block, left=of D] (loss) {Loss Compute};

    \draw[arr] (ct) -- (G);
    \draw[arr] (G) -- (fake);
    \draw[arr] (fake) -- (D);
    \draw[arr] (real) -- (D);
    \draw[arr] (ct) |- (D);
    \draw[arr] (D) -- (loss);
    \draw[arr] (fake) -| (loss);
    \draw[arr] (real) -| (loss);
\end{tikzpicture}
}
\caption{Pix2Pix cGAN Training: Discriminator sees (CT, Real) vs (CT, Fake).}
\end{figure}

\textbf{Loss Function:} The total loss is $L = L_{cGAN} + \lambda L_1$. The $L_1$ loss encourages pixel-wise similarity to the ground truth, while the GAN loss pushes the generator to produce realistic textures.

\section{INFERENCE PIPELINE}
Inference is a direct feedforward pass through the generator.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=10mm,
    block/.style={draw=brand, thick, fill=softbg, minimum width=2.5cm, minimum height=0.8cm, align=center, rounded corners},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[block] (in) {Patient CT};
    \node[block, right=of in] (p) {Preprocess};
    \node[block, right=of p] (g) {U-Net Inf};
    \node[block, right=of g] (o) {Synthetic MRI};

    \draw[arr] (in) -- (p);
    \draw[arr] (p) -- (g);
    \draw[arr] (g) -- (o);
\end{tikzpicture}
\caption{Pix2Pix linear inference flow.}
\end{figure}
""" % {"region": region, "MODEL": "Pix2Pix"}

def get_diffusion_content(region, paired=True):
    model_type = "Paired Diffusion" if paired else "Unpaired Diffusion"
    return r"""
\section{ARCHITECTURAL OVERVIEW}
This model utilizes a Latent Diffusion Framework (specifically SD-Turbo) adapted for medical imaging via LoRA (Low-Rank Adaptation). It learns to predict and remove noise in a compressed latent space.

\subsection{Architecture (SD-Turbo + LoRA)}
The core is a U-Net noise predictor conditioned on the input CT slice.

\begin{figure}[H]
\centering
\resizebox{0.9\textwidth}{!}{
\begin{tikzpicture}[
    node distance=10mm,
    block/.style={draw=brand, thick, fill=softbg, minimum width=2.2cm, minimum height=0.8cm, align=center, rounded corners},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[block] (ct) {Input CT};
    \node[block, right=of ct] (enc) {VAE Encoder\\(Latent $z$)};
    \node[block, right=of enc] (unet) {U-Net\\(LoRA Adapted)};
    \node[block, right=of unet] (dec) {VAE Decoder\\(Image Space)};
    \node[block, right=of dec] (out) {MRI Target};
    
    \node[block, below=of unet] (t) {Timestep $t$};
    \node[block, right=of t] (cond) {Conditioning\\(CT Embed)};

    \draw[arr] (ct) -- (enc);
    \draw[arr] (enc) -- (unet);
    \draw[arr] (unet) -- (dec);
    \draw[arr] (dec) -- (out);
    \draw[arr] (t) -- (unet);
    \draw[arr] (cond) -- (unet);
\end{tikzpicture}
}
\caption{Diffusion architecture using pre-trained VAE and LoRA-adapted U-Net.}
\end{figure}

\textbf{Latent Representation:} By operating in the latent space of a pre-trained VAE, the model reduces computational requirements while benefiting from the robust feature representations of stable diffusion.

\textbf{LoRA Adaptation:} Instead of fine-tuning the entire U-Net, we only update low-rank matrices in the attention layers. This preserves the generative prior while specializing the model for %(region)s CT-MRI translation.

\section{TRAINING AND INFERENCE}
The model is trained using a denoising objective.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=10mm,
    block/.style={draw=brand, thick, fill=softbg, minimum width=2.5cm, minimum height=0.8cm, align=center, rounded corners},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[block] (n) {Noise $\epsilon$};
    \node[block, right=of n] (u) {U-Net Predictor};
    \node[block, right=of u] (d) {Denoising Step};
    \node[block, right=of d] (o) {Recon Image};

    \draw[arr] (n) -- (u);
    \draw[arr] (u) -- (d);
    \draw[arr] (d) -- (o);
    \draw[arr, dashed] (o.south) .. controls +(0,-1) and +(0,-1) .. (u.south);
\end{tikzpicture}
\caption{Iterative denoising (or 1-step for SD-Turbo).}
\end{figure}

\textbf{One-Step Synthesis:} Thanks to SD-Turbo's adversarial distillation, the model can generate high-fidelity MRI slices in a single forward pass, making it nearly as fast as GAN-based models during inference.
""" % {"region": region, "MODEL": model_type}

def get_comparison_content():
    return r"""
\section{ARCHITECTURAL COMPARISON}
We compare the performance of 8 canonical models across two anatomical regions (Brain and Pelvis).

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=1.5cm and 2cm,
    box/.style={draw=brand, thick, rounded corners, align=center, fill=softbg, minimum width=3.5cm, minimum height=1cm},
    arr/.style={-{Latex[length=2mm]}, thick, draw=accent}
]
    \node[box, fill=brand!10] (root) {Medical Image Translation};
    \node[box] (gan) [below left=of root] {GAN Models\\(2D Convolutional)};
    \node[box] (diff) [below right=of root] {Diffusion Models\\(2.5D Latent Space)};
    
    \node[box] (pix) [below=of gan] {Pix2Pix\\(Paired)};
    \node[box] (cyc) [left=1cm of pix] {CycleGAN\\(Unpaired)};
    
    \node[box] (pdiff) [below=of diff] {Paired Diffusion\\(SD-Turbo)};
    \node[box] (udiff) [right=1cm of pdiff] {Unpaired Diffusion\\(Cycle Latent)};

    \draw[arr] (root) -- (gan);
    \draw[arr] (root) -- (diff);
    \draw[arr] (gan) -- (pix);
    \draw[arr] (gan) -- (cyc);
    \draw[arr] (diff) -- (pdiff);
    \draw[arr] (diff) -- (udiff);
\end{tikzpicture}
\caption{Taxonomy of Implemented Model Architectures.}
\end{figure}

\section{MODEL PERFORMANCE SUMMARY}
\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{lYYYY}
\toprule
\textbf{Feature} & \textbf{CycleGAN} & \textbf{Pix2Pix} & \textbf{Paired Diff} & \textbf{Unpaired Diff} \\
\midrule
Supervision & Unpaired & Paired & Paired & Unpaired \\
Architecture & ResNet-9 & U-Net-256 & LDM + LoRA & Cycle LDM \\
Resolution & $256^2$ & $256^2$ & $512^2$ & $512^2$ \\
Dimensionality & 2D & 2D & 2.5D & 2.5D \\
Inference & Direct & Direct & 1-Step & 1-Step \\
\bottomrule
\end{tabularx}
\caption{Comparison of architectural specifications across model families.}
\end{table}

\section{TECHNICAL DISCUSSION}
The paired models (Pix2Pix and Paired Diffusion) exhibit higher structural fidelity in regions with high alignment (Brain). However, CycleGAN and Unpaired Diffusion demonstrate superior robustness to misaligned clinical data, which is frequent in Pelvis datasets. The use of 2.5D representations in Diffusion models significantly reduces slice-to-slice artifacts compared to purely 2D GAN models.
"""

# -------------------- Execution Logic --------------------

MODELS = [
    ("cyclegan_brain", "CycleGAN (Brain)", "Brain", r"Unpaired (CT $\leftrightarrow$ MRI)", "CycleGAN"),
    ("cyclegan_pelvis", "CycleGAN (Pelvis)", "Pelvis", r"Unpaired (CT $\leftrightarrow$ MRI)", "CycleGAN"),
    ("pix2pix_brain", "Pix2Pix (Brain)", "Brain", "Paired (CT $\to$ MRI)", "Pix2Pix"),
    ("pix2pix_pelvis", "Pix2Pix (Pelvis)", "Pelvis", "Paired (CT $\to$ MRI)", "Pix2Pix"),
    ("paired_diffusion_brain", "Paired Diffusion (Brain)", "Brain", "Paired (CT $\to$ MRI)", "Paired Diffusion"),
    ("paired_diffusion_pelvis", "Paired Diffusion (Pelvis)", "Pelvis", "Paired (CT $\to$ MRI)", "Paired Diffusion"),
    ("unpaired_diffusion_brain", "Unpaired Diffusion (Brain)", "Brain", r"Unpaired (CT $\leftrightarrow$ MRI)", "Unpaired Diffusion"),
    ("unpaired_diffusion_pelvis", "Unpaired Diffusion (Pelvis)", "Pelvis", r"Unpaired (CT $\leftrightarrow$ MRI)", "Unpaired Diffusion")
]

def main():
    # 1. Process individual model reports
    for folder, model_title, region, modality, m_type in MODELS:
        tex_path = os.path.join(DOCS_DIR, folder, f"{folder}_report.tex")
        
        # Build Header
        header = PREAMBLE % {"model_name": model_title, "model_title": model_title, "region": region, "modality": modality}
        
        # Build Content
        if m_type == "CycleGAN":
            content = get_cyclegan_content(region)
        elif m_type == "Pix2Pix":
            content = get_pix2pix_content(region)
        elif "Diffusion" in m_type:
            paired = "Paired" in m_type
            content = get_diffusion_content(region, paired)
        
        full_tex = header + content + POSTAMBLE
        
        with open(tex_path, "w") as f:
            f.write(full_tex)
            
        # Recompile
        subprocess.run([TECTONIC_BIN, f"{folder}_report.tex"], cwd=os.path.join(DOCS_DIR, folder))
        print(f"Polished and Recompiled {folder}")

    # 2. Process comparison report
    comp_folder = "all_models_comparison"
    comp_tex = os.path.join(DOCS_DIR, comp_folder, "all_models_comparison_report.tex")
    comp_header = PREAMBLE % {"model_name": "Comparative Analysis", "model_title": "Multi-Model Comparative Analysis", "region": "Brain/Pelvis", "modality": "Mixed"}
    comp_content = get_comparison_content()
    full_comp_tex = comp_header + comp_content + POSTAMBLE
    
    with open(comp_tex, "w") as f:
        f.write(full_comp_tex)
        
    subprocess.run([TECTONIC_BIN, "all_models_comparison_report.tex"], cwd=os.path.join(DOCS_DIR, comp_folder))
    print("Polished and Recompiled all_models_comparison")

if __name__ == "__main__":
    main()

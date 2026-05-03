import os
import re
import subprocess

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
TECTONIC_BIN = os.environ.get("TECTONIC_BIN", "tectonic")

MODELS = [
    ("cyclegan_brain", "CycleGAN", "Brain"),
    ("cyclegan_pelvis", "CycleGAN", "Pelvis"),
    ("pix2pix_brain", "Pix2Pix", "Brain"),
    ("pix2pix_pelvis", "Pix2Pix", "Pelvis"),
    ("paired_diffusion_brain", "Paired Diffusion", "Brain"),
    ("paired_diffusion_pelvis", "Paired Diffusion", "Pelvis"),
    ("unpaired_diffusion_brain", "Unpaired Diffusion", "Brain"),
    ("unpaired_diffusion_pelvis", "Unpaired Diffusion", "Pelvis")
]

CYCLEGAN_TIKZ = r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=1.5cm and 2cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=8pt, minimum width=3.5cm, minimum height=1cm},
  lbl/.style={font=\sffamily\small\bfseries, color=darkblue},
  loss/.style={draw=red!80!black, thick, rounded corners, align=center, fill=red!5, font=\sffamily\small\bfseries, inner sep=6pt},
  arr/.style={->, >=stealth, thick, darkblue}
]
  \node[lbl] (g_lbl) at (0, 0) {Generators ($G$ and $F$)};
  
  \node[box] (ct_in) [below left=0.5cm and 0.5cm of g_lbl] {2D {REGION} CT Input};
  \node[box] (g_gen) [below=0.8cm of ct_in] {ResNet Generator $G$\\ (CT $\rightarrow$ MRI)};
  \node[box] (mri_out) [below=0.8cm of g_gen] {Fake {REGION} MRI};
  
  \node[box] (mri_in) [below right=0.5cm and 0.5cm of g_lbl] {2D {REGION} MRI Input};
  \node[box] (f_gen) [below=0.8cm of mri_in] {ResNet Generator $F$\\ (MRI $\rightarrow$ CT)};
  \node[box] (ct_out) [below=0.8cm of f_gen] {Fake {REGION} CT};

  \draw[arr] (ct_in) -- (g_gen);
  \draw[arr] (g_gen) -- (mri_out);
  
  \draw[arr] (mri_in) -- (f_gen);
  \draw[arr] (f_gen) -- (ct_out);

  \node[lbl] (d_lbl) [below=1.0cm of g_lbl |- mri_out] {Discriminators ($D_{\text{MRI}}$ and $D_{\text{CT}}$)};
  
  \node[box] (dmri) [below left=0.5cm and 0.5cm of d_lbl] {PatchGAN $D_{\text{MRI}}$};
  \node[box] (real_mri) [left=1cm of dmri] {Real {REGION} MRI};
  
  \node[box] (dct) [below right=0.5cm and 0.5cm of d_lbl] {PatchGAN $D_{\text{CT}}$};
  \node[box] (real_ct) [right=1cm of dct] {Real {REGION} CT};

  \draw[arr] (mri_out) -- (dmri);
  \draw[arr] (real_mri) -- (dmri);
  
  \draw[arr] (ct_out) -- (dct);
  \draw[arr] (real_ct) -- (dct);
\end{tikzpicture}
\caption{{MODEL} Architecture Overview ({REGION})}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=1.5cm and 2cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=6pt, minimum width=3.5cm},
  loss/.style={draw=red!80!black, thick, rounded corners, align=center, fill=red!5, font=\sffamily\small\bfseries, inner sep=6pt},
  arr/.style={->, >=stealth, thick, darkblue}
]
  \node[box] (ct) at (-3, 0) {Real CT};
  \node[box] (g) [below=1cm of ct] {Generator $G$};
  \node[box] (fake_mri) [below=1cm of g] {Fake MRI};
  \node[box] (f_cyc) [below=1cm of fake_mri] {Generator $F$};
  \node[box] (rec_ct) [below=1cm of f_cyc] {Recon CT};
  
  \node[box] (mri) at (3, 0) {Real MRI};
  \node[box] (f) [below=1cm of mri] {Generator $F$};
  \node[box] (fake_ct) [below=1cm of f] {Fake CT};
  \node[box] (g_cyc) [below=1cm of fake_ct] {Generator $G$};
  \node[box] (rec_mri) [below=1cm of g_cyc] {Recon MRI};

  \draw[arr] (ct) -- (g); \draw[arr] (g) -- (fake_mri); \draw[arr] (fake_mri) -- (f_cyc); \draw[arr] (f_cyc) -- (rec_ct);
  \draw[arr] (mri) -- (f); \draw[arr] (f) -- (fake_ct); \draw[arr] (fake_ct) -- (g_cyc); \draw[arr] (g_cyc) -- (rec_mri);

  \node[loss] (cyc_loss) at (0, -6) {Cycle L1 Loss};
  \draw[arr, dashed, red!80!black] (ct.east) -- (cyc_loss.north);
  \draw[arr, dashed, red!80!black] (rec_ct.east) -- (cyc_loss.south);
  \draw[arr, dashed, red!80!black] (mri.west) -- (cyc_loss.north);
  \draw[arr, dashed, red!80!black] (rec_mri.west) -- (cyc_loss.south);

  \node[loss] (adv_loss) at (0, -2.5) {Adversarial Loss};
  \draw[arr, dashed, red!80!black] (fake_mri.east) -- (adv_loss.west);
  \draw[arr, dashed, red!80!black] (fake_ct.west) -- (adv_loss.east);
\end{tikzpicture}
\caption{{MODEL} Training Workflow and Loss Computations ({REGION})}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=0.8cm and 1.5cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=6pt, minimum height=0.8cm},
  arr/.style={->, >=stealth, thick, darkblue}
]
  \node[box] (ct_cohort) at (-3, 0) {Unpaired CT Cohort};
  \node[box] (slicer1) [below=of ct_cohort] {Extract 2D CT};
  \node[box] (norm1) [below=of slicer1] {Norm to $[-1,1]$};
  \node[box] (g) [below=of norm1] {$G$: CT $\rightarrow$ MRI};
  \node[box] (fake_mri) [below=of g] {Fake MRI};
  
  \node[box] (mri_cohort) at (3, 0) {Unpaired MRI Cohort};
  \node[box] (slicer2) [below=of mri_cohort] {Extract 2D MRI};
  \node[box] (norm2) [below=of slicer2] {Norm to $[-1,1]$};
  \node[box] (f) [below=of norm2] {$F$: MRI $\rightarrow$ CT};
  \node[box] (fake_ct) [below=of f] {Fake CT};

  \draw[arr] (ct_cohort) -- (slicer1); \draw[arr] (slicer1) -- (norm1); \draw[arr] (norm1) -- (g); \draw[arr] (g) -- (fake_mri);
  \draw[arr] (mri_cohort) -- (slicer2); \draw[arr] (slicer2) -- (norm2); \draw[arr] (norm2) -- (f); \draw[arr] (f) -- (fake_ct);
\end{tikzpicture}
\caption{{MODEL} Data Flow ({REGION})}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=0.8cm and 1.5cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=6pt, minimum height=0.8cm},
  arr/.style={->, >=stealth, thick, darkblue}
]
  \node[box] (input) at (0, 0) {New {REGION} CT Volume};
  \node[box] (slicer) [below=of input] {Sequential 2D Slicing};
  \node[box] (norm) [below=of slicer] {Min-Max Norm $[-1,1]$};
  \node[box] (gen) [below=of norm] {Trained ResNet Generator $G$};
  \node[box] (out) [below=of gen] {Fake MRI Slices};
  \node[box] (stack) [below=of out] {Stack into 3D NIfTI};

  \draw[arr] (input) -- (slicer); \draw[arr] (slicer) -- (norm); \draw[arr] (norm) -- (gen); \draw[arr] (gen) -- (out); \draw[arr] (out) -- (stack);
\end{tikzpicture}
\caption{{MODEL} Inference Pipeline ({REGION})}
\end{figure}
"""

PIX2PIX_TIKZ = r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=1.5cm and 2cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=8pt, minimum width=3.5cm, minimum height=1cm},
  lbl/.style={font=\sffamily\small\bfseries, color=darkblue},
  loss/.style={draw=red!80!black, thick, rounded corners, align=center, fill=red!5, font=\sffamily\small\bfseries, inner sep=6pt},
  arr/.style={->, >=stealth, thick, darkblue}
]
  \node[box] (ct_in) at (-3, 0) {2D {REGION} CT Input};
  \node[box] (g_gen) [below=1cm of ct_in] {U-Net Generator $G$};
  \node[box] (mri_out) [below=1cm of g_gen] {Fake {REGION} MRI};
  
  \node[box] (mri_real) at (3, 0) {Real {REGION} MRI};
  
  \node[box] (dmri) [below=2cm of mri_real] {PatchGAN Discriminator $D$};

  \draw[arr] (ct_in) -- (g_gen);
  \draw[arr] (g_gen) -- (mri_out);
  
  \draw[arr] (mri_out.east) -- (dmri.west);
  \draw[arr] (mri_real.south) -- (dmri.north);
  \draw[arr, dashed] (ct_in.east) -- (dmri.north west);

  \node[loss] (l1_loss) at (0, -2.2) {L1 Loss};
  \draw[arr, dashed, red!80!black] (mri_real.south west) -- (l1_loss.north east);
  \draw[arr, dashed, red!80!black] (mri_out.north east) -- (l1_loss.south west);

\end{tikzpicture}
\caption{{MODEL} Architecture and Training Overview ({REGION})}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=0.8cm and 1.5cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=6pt, minimum height=0.8cm},
  arr/.style={->, >=stealth, thick, darkblue}
]
  \node[box] (input) at (0, 0) {New {REGION} CT Volume};
  \node[box] (slicer) [below=of input] {Sequential 2D Slicing};
  \node[box] (norm) [below=of slicer] {Min-Max Norm $[-1,1]$};
  \node[box] (gen) [below=of norm] {Trained U-Net Generator};
  \node[box] (out) [below=of gen] {Fake MRI Slices};
  \node[box] (stack) [below=of out] {Stack into 3D NIfTI};

  \draw[arr] (input) -- (slicer); \draw[arr] (slicer) -- (norm); \draw[arr] (norm) -- (gen); \draw[arr] (gen) -- (out); \draw[arr] (out) -- (stack);
\end{tikzpicture}
\caption{{MODEL} Inference Pipeline ({REGION})}
\end{figure}
"""

DIFFUSION_TIKZ = r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=1.5cm and 2cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=8pt, minimum width=3.5cm, minimum height=1cm},
  lbl/.style={font=\sffamily\small\bfseries, color=darkblue},
  loss/.style={draw=red!80!black, thick, rounded corners, align=center, fill=red!5, font=\sffamily\small\bfseries, inner sep=6pt},
  arr/.style={->, >=stealth, thick, darkblue}
]
  \node[box] (x0) at (-4, 0) {Clean {REGION} Image $x_0$};
  \node[box] (noise) [below=1.5cm of x0] {Add Gaussian Noise\\ (Forward Process)};
  \node[box] (xt) [below=1.5cm of noise] {Noisy Image $x_t$};
  
  \node[box] (unet) at (4, -3) {U-Net Noise Predictor\\ $\epsilon_\theta(x_t, t)$};
  
  \node[box] (pred_noise) [above=1.5cm of unet] {Predicted Noise $\hat{\epsilon}$};
  \node[box] (true_noise) [above=1.5cm of pred_noise] {True Noise $\epsilon$};

  \draw[arr] (x0) -- (noise);
  \draw[arr] (noise) -- (xt);
  
  \draw[arr] (xt) -- (unet);
  \draw[arr] (unet) -- (pred_noise);
  
  \node[loss] (mse) at (0, 0) {MSE Loss};
  \draw[arr, dashed, red!80!black] (true_noise) -- (mse);
  \draw[arr, dashed, red!80!black] (pred_noise) -- (mse);

\end{tikzpicture}
\caption{{MODEL} Training and Forward Process ({REGION})}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=0.8cm and 1.5cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=6pt, minimum height=0.8cm},
  arr/.style={->, >=stealth, thick, darkblue}
]
  \node[box] (noise) at (0, 0) {Pure Gaussian Noise $x_T$};
  \node[box] (unet) [below=of noise] {Trained U-Net Predicts Noise};
  \node[box] (denoise) [below=of unet] {Remove Noise Step $t \rightarrow t-1$};
  \node[box] (loop) [right=of unet] {Iterate $T$ Steps};
  \node[box] (final) [below=of denoise] {Final Generated Image $x_0$};

  \draw[arr] (noise) -- (unet); 
  \draw[arr] (unet) -- (denoise); 
  \draw[arr] (denoise.east) -- (loop.south);
  \draw[arr] (loop.north) -- (unet.east);
  \draw[arr] (denoise) -- (final);
\end{tikzpicture}
\caption{{MODEL} Inference and Reverse Denoising ({REGION})}
\end{figure}
"""

def main():
    for folder, m_type, region in MODELS:
        tex_path = os.path.join(DOCS_DIR, folder, f"{folder}_report.tex")
        if not os.path.exists(tex_path):
            continue
            
        with open(tex_path, "r") as f:
            content = f.read()
            
        # Strip from Required Diagrams onwards
        parts = re.split(r'\\subsection\*\{Required Diagrams\}', content)
        if len(parts) == 1:
            # Maybe the section name was altered, try looking for Figure 1 or Architecture
            parts = re.split(r'\\subsubsection\*\{.*?Architecture Diagram\}', content)
            if len(parts) == 1:
                parts = re.split(r'\\begin\{tikzpicture\}', content)
                # Keep everything before the first tikzpicture, but remove \begin{figure}
                if len(parts) > 1:
                    content = parts[0][:parts[0].rfind(r'\begin{figure}')]
        else:
            content = parts[0]
            
        # Clean up trailing newlines
        content = content.strip() + "\n\n"
        
        # Inject new templates based on model type
        if "CycleGAN" in m_type:
            tikz = CYCLEGAN_TIKZ
        elif "Pix2Pix" in m_type:
            tikz = PIX2PIX_TIKZ
        else:
            tikz = DIFFUSION_TIKZ
            
        tikz = tikz.replace("{REGION}", region).replace("{MODEL}", m_type)
        
        new_content = content + tikz + "\n\\end{document}\n"
        
        with open(tex_path, "w") as f:
            f.write(new_content)
            
        # Compile
        try:
            subprocess.run([TECTONIC_BIN, f"{folder}_report.tex"], cwd=os.path.join(DOCS_DIR, folder), check=True, capture_output=True)
            print(f"Compiled {folder}")
        except subprocess.CalledProcessError as e:
            print(f"Failed {folder}: {e.stderr.decode()}")

if __name__ == "__main__":
    main()

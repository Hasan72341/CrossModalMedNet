import os
import re
import subprocess

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
TECTONIC_BIN = os.environ.get("TECTONIC_BIN", "tectonic")
FILES_TO_PROCESS = ["all_models_comparison.md"]

LATEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{geometry}
\geometry{letterpaper, margin=1.2in}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{listings}
\usepackage{caption}

\definecolor{darkblue}{RGB}{20, 40, 100}
\usepackage{titlesec}
\titleformat*{\section}{\Large\sffamily\bfseries\color{darkblue}}
\titleformat*{\subsection}{\large\sffamily\bfseries\color{darkblue!85}}
\titleformat*{\subsubsection}{\normalsize\sffamily\bfseries\color{darkblue!85}}

\usepackage{tocloft}
\renewcommand{\cftsecfont}{\sffamily\bfseries\color{darkblue}}
\renewcommand{\cftsubsecfont}{\sffamily\color{darkblue!85}}
\renewcommand{\contentsname}{\sffamily\bfseries\color{darkblue} Contents}

\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit, backgrounds}

\captionsetup{labelfont=bf, textfont=it, skip=10pt}

\title{\vspace{-2cm}\sffamily\bfseries\color{darkblue} %(title)s}
\author{\sffamily Project Group 5}
\date{}

\begin{document}
\maketitle
\tableofcontents
\vspace{1cm}
\hrule
\vspace{1cm}

"""

LATEX_POSTAMBLE = r"""
\end{document}
"""

def main():
    folder_name = "all_models_comparison"
    target_dir = os.path.join(DOCS_DIR, folder_name)
    md_path = os.path.join(target_dir, "all_models_comparison.md")
    report_tex = os.path.join(target_dir, f"{folder_name}_report.tex")
    
    with open(md_path, "r") as f:
        content = f.read()
        
    # Standard Markdown to LaTeX converter specific for this document
    title = "Comparative Summary of Medical Translation Models"
    
    tex_parts = []
    
    lines = content.split('\n')
    in_table = False
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip the title in markdown since we put it in preamble
        if line.startswith("# Comparative Summary"):
            i += 1
            continue
            
        if line.startswith("```mermaid"):
            # custom tikz for taxonomy
            tikz = r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=1.5cm and 2cm,
  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=8pt, minimum width=3.5cm, minimum height=1cm},
  lbl/.style={font=\sffamily\small\bfseries, color=darkblue},
  arr/.style={-, thick, darkblue}
]
  \node[box, fill=darkblue!15, font=\sffamily\large\bfseries] (root) at (0, 0) {Medical Image Translation Models};
  
  \node[box, fill=darkblue!10] (gan) at (-4, -2) {GAN Paradigm\\(2D Convolutional)};
  \node[box, fill=darkblue!10] (diff) at (4, -2) {Diffusion Paradigm\\(2.5D Latent SD-Turbo)};
  
  \node[box] (pix) at (-6, -4) {Pix2Pix\\(Paired L1 + GAN)};
  \node[box] (cyc) at (-2, -4) {CycleGAN\\(Unpaired Cycle-Consistency)};
  
  \node[box] (pdiff) at (2, -4) {Paired Diffusion\\(LoRA UNet + Skips)};
  \node[box] (udiff) at (6, -4) {Unpaired Diffusion\\(Dual UNet + Skips)};
  
  \draw[arr] (root) -- (0, -1) -| (gan);
  \draw[arr] (root) -- (0, -1) -| (diff);
  
  \draw[arr] (gan) -- (-4, -3) -| (pix);
  \draw[arr] (gan) -- (-4, -3) -| (cyc);
  
  \draw[arr] (diff) -- (4, -3) -| (pdiff);
  \draw[arr] (diff) -- (4, -3) -| (udiff);
\end{tikzpicture}
\caption{Architectural Taxonomy of Implemented Translation Models}
\end{figure}
            """
            tex_parts.append(tikz)
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                i += 1
            i += 1
            continue
            
        if line.startswith('|'):
            if not in_table:
                in_table = True
                tex_parts.append(r"\begin{center}")
                tex_parts.append(r"\begin{tabular}{lllll}")
                tex_parts.append(r"\toprule")
            if '---' in line:
                tex_parts.append(r"\midrule")
                i += 1
                continue
            row = line.strip('|').split('|')
            row_tex = []
            for cell in row:
                cell = cell.strip().replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
                cell = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', cell)
                row_tex.append(cell)
            tex_parts.append(" & ".join(row_tex) + r" \\")
            i += 1
            continue
        else:
            if in_table:
                in_table = False
                tex_parts.append(r"\bottomrule")
                tex_parts.append(r"\end{tabular}")
                tex_parts.append(r"\end{center}")

        m_h3 = re.match(r'^###\s+(.*)', line)
        if m_h3:
            tex_parts.append(r"\subsubsection*{" + m_h3.group(1).replace('_', r'\_').replace('&', r'\&') + "}")
            i += 1
            continue
        m_h2 = re.match(r'^##\s+(.*)', line)
        if m_h2:
            tex_parts.append(r"\subsection*{" + m_h2.group(1).replace('_', r'\_').replace('&', r'\&') + "}")
            i += 1
            continue
        m_h1 = re.match(r'^#\s+(.*)', line)
        if m_h1:
            tex_parts.append(r"\section*{" + m_h1.group(1).replace('_', r'\_').replace('&', r'\&') + "}")
            i += 1
            continue

        m_list = re.match(r'^-\s+(.*)', line)
        if m_list:
            tex_parts.append(r"\begin{itemize}")
            while i < len(lines) and lines[i].strip().startswith('-'):
                content = lines[i].strip()[1:].strip()
                content = content.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
                content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
                content = re.sub(r'`(.*?)`', r'\\texttt{\1}', content)
                content = content.replace('_', r'\_')
                tex_parts.append(r"\item " + content)
                i += 1
            tex_parts.append(r"\end{itemize}")
            continue
            
        if line == '':
            tex_parts.append("")
        elif line == '---':
            tex_parts.append(r"\vspace{0.5cm}\hrule\vspace{0.5cm}")
        else:
            content = line
            content = content.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
            content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
            content = re.sub(r'`(.*?)`', r'\\texttt{\1}', content)
            content = content.replace('_', r'\_')
            tex_parts.append(content)
            
        i += 1
            
    if in_table:
        tex_parts.append(r"\bottomrule")
        tex_parts.append(r"\end{tabular}")
        tex_parts.append(r"\end{center}")
        
    body = "\n".join(tex_parts)
    body = body.replace('→', r'$\rightarrow$').replace('↔', r'$\leftrightarrow$').replace('—', r'---')
    body = body.replace(r'\$CT \textbackslash{}rightarrow Fake MRI \textbackslash{}rightarrow Reconstructed CT\$', r'$CT \rightarrow Fake MRI \rightarrow Reconstructed CT$')
    
    with open(report_tex, "w") as f:
        f.write(LATEX_PREAMBLE % {"title": title})
        f.write(body)
        f.write(LATEX_POSTAMBLE)
        
    print(f"Generated {report_tex}")
    
    try:
        subprocess.run([TECTONIC_BIN, f"{folder_name}_report.tex"], cwd=target_dir, check=True, capture_output=True)
        print("Successfully compiled PDF")
    except subprocess.CalledProcessError as e:
        print(f"tectonic error: {e.stderr.decode()}")

if __name__ == "__main__":
    main()

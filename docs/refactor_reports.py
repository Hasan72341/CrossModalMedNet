import os
import shutil
import subprocess
import re

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
TECTONIC_BIN = os.environ.get("TECTONIC_BIN", "tectonic")

FILES_TO_PROCESS = [
    "cyclegan_brain_documentation.md",
    "cyclegan_pelvis_documentation.md",
    "paired_diffusion_brain_documentation.md",
    "paired_diffusion_pelvis_documentation.md",
    "pix2pix_brain_documentation.md",
    "pix2pix_pelvis_documentation.md",
    "unpaired_diffusion_brain_documentation.md",
    "unpaired_diffusion_pelvis_documentation.md",
    "all_models_comparison.md"
]

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

def mermaid_to_tikz(mermaid_code):
    nodes = {}
    edges = []
    
    for line in mermaid_code.split('\n'):
        line = line.strip()
        if not line or line.startswith('graph') or line.startswith('flowchart') or line.startswith('subgraph') or line.startswith('end'):
            continue
            
        m = re.match(r'^([A-Za-z0-9_]+)\[(.*?)\]\s*-->\s*([A-Za-z0-9_]+)\[(.*?)\]$', line)
        if m:
            n1, t1, n2, t2 = m.groups()
            nodes[n1] = t1.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
            nodes[n2] = t2.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
            edges.append((n1, n2))
            continue
            
        m = re.match(r'^([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)\[(.*?)\]$', line)
        if m:
            n1, n2, t2 = m.groups()
            if n1 not in nodes: nodes[n1] = n1
            nodes[n2] = t2.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
            edges.append((n1, n2))
            continue
            
        m = re.match(r'^([A-Za-z0-9_]+)\[(.*?)\]\s*-->\s*([A-Za-z0-9_]+)$', line)
        if m:
            n1, t1, n2 = m.groups()
            nodes[n1] = t1.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
            if n2 not in nodes: nodes[n2] = n2
            edges.append((n1, n2))
            continue
            
        m = re.match(r'^([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)$', line)
        if m:
            n1, n2 = m.groups()
            if n1 not in nodes: nodes[n1] = n1
            if n2 not in nodes: nodes[n2] = n2
            edges.append((n1, n2))
            continue

    depths = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        for u, v in edges:
            if depths[u] + 1 > depths[v]:
                depths[v] = depths[u] + 1
                
    levels = {}
    for n, d in depths.items():
        if d not in levels: levels[d] = []
        levels[d].append(n)
        
    tikz = []
    tikz.append(r"\begin{figure}[h!]")
    tikz.append(r"\centering")
    tikz.append(r"\begin{tikzpicture}[")
    tikz.append(r"  box/.style={draw=darkblue, thick, rounded corners, align=center, fill=darkblue!5, font=\sffamily\small, inner sep=6pt, minimum height=0.8cm},")
    tikz.append(r"  arr/.style={->, >=stealth, thick, darkblue}")
    tikz.append(r"]")
    
    for d in sorted(levels.keys()):
        level_nodes = levels[d]
        n_count = len(level_nodes)
        y = -d * 1.5
        for i, n in enumerate(level_nodes):
            x = (i - (n_count - 1) / 2.0) * 4.0
            tikz.append(rf"  \node[box] ({n}) at ({x}, {y}) {{{nodes[n]}}};")
            
    for u, v in edges:
        tikz.append(rf"  \draw[arr] ({u}) -- ({v});")
        
    tikz.append(r"\end{tikzpicture}")
    tikz.append(r"\vspace{0.5cm}")
    tikz.append(r"\end{figure}")
    
    return '\n'.join(tikz)

def fallback_markdown_to_latex(md_content):
    title = "Model Documentation"
    m = re.search(r'^#\s+(.*)', md_content, flags=re.MULTILINE)
    if m:
        title = m.group(1).strip()
        md_content = re.sub(r'^#\s+.*$', '', md_content, count=1, flags=re.MULTILINE)

    blocks = re.split(r'(```.*?```)', md_content, flags=re.DOTALL)
    tex_parts = []
    
    for block in blocks:
        if block.startswith('```'):
            lines = block.split('\n')
            lang = lines[0].replace('`', '').strip()
            code = '\n'.join(lines[1:-1])
            
            if lang == 'mermaid':
                tex_parts.append(mermaid_to_tikz(code))
            else:
                tex_parts.append(r"\begin{lstlisting}")
                tex_parts.append(code)
                tex_parts.append(r"\end{lstlisting}")
            continue
            
        lines = block.split('\n')
        in_list = False
        in_table = False
        parsed_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith('|'):
                if not in_table:
                    in_table = True
                    parsed_lines.append(r"\begin{center}")
                    parsed_lines.append(r"\begin{tabular}{lllll}")
                    parsed_lines.append(r"\toprule")
                if '---' in line_stripped:
                    parsed_lines.append(r"\midrule")
                    continue
                row = line_stripped.strip('|').split('|')
                row_tex = []
                for cell in row:
                    cell = cell.strip().replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
                    cell = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', cell)
                    row_tex.append(cell)
                parsed_lines.append(" & ".join(row_tex) + r" \\")
                continue
            else:
                if in_table:
                    in_table = False
                    parsed_lines.append(r"\bottomrule")
                    parsed_lines.append(r"\end{tabular}")
                    parsed_lines.append(r"\end{center}")

            m_h3 = re.match(r'^###\s+(.*)', line_stripped)
            if m_h3:
                parsed_lines.append(r"\subsubsection*{" + m_h3.group(1).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#') + "}")
                continue
            m_h2 = re.match(r'^##\s+(.*)', line_stripped)
            if m_h2:
                parsed_lines.append(r"\subsection*{" + m_h2.group(1).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#') + "}")
                continue
            m_h1 = re.match(r'^#\s+(.*)', line_stripped)
            if m_h1:
                parsed_lines.append(r"\section*{" + m_h1.group(1).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#') + "}")
                continue

            m_list = re.match(r'^-\s+(.*)', line_stripped)
            if m_list:
                if not in_list:
                    in_list = True
                    parsed_lines.append(r"\begin{itemize}")
                content = m_list.group(1)
                content = content.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
                content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
                content = re.sub(r'`(.*?)`', r'\\texttt{\1}', content)
                content = re.sub(r'(?<!\$)\b_\b', r'\\_', content)
                content = content.replace('_', r'\_')
                parsed_lines.append(r"\item " + content)
                continue
            else:
                if in_list and line_stripped == '':
                    in_list = False
                    parsed_lines.append(r"\end{itemize}")
            
            if line_stripped == '':
                parsed_lines.append("")
            else:
                content = line_stripped
                content = content.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
                content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
                content = re.sub(r'`(.*?)`', r'\\texttt{\1}', content)
                content = content.replace('_', r'\_')
                parsed_lines.append(content)
                
        if in_list:
            parsed_lines.append(r"\end{itemize}")
        if in_table:
            parsed_lines.append(r"\bottomrule")
            parsed_lines.append(r"\end{tabular}")
            parsed_lines.append(r"\end{center}")
            
        tex_parts.append('\n'.join(parsed_lines))
        
    body = "\n".join(tex_parts)
    body = body.replace('→', r'$\rightarrow$').replace('↔', r'$\leftrightarrow$').replace('—', r'---')
    body = body.replace(r'\_\_', r'_')
    body = body.replace(r'\$F(G(CT)) \textbackslash{}approx CT\$', r'$F(G(CT)) \approx CT$')
    body = body.replace(r'\$G(F(MRI)) \textbackslash{}approx MRI\$', r'$G(F(MRI)) \approx MRI$')


    return title, body

def main():
    with open(os.path.join(DOCS_DIR, "refactor_log.txt"), "w") as log:
        for md_file in FILES_TO_PROCESS:
            folder_name = md_file.replace("_documentation.md", "").replace(".md", "")
            target_dir = os.path.join(DOCS_DIR, folder_name)
            md_path = os.path.join(target_dir, md_file)
            
            if not os.path.exists(md_path):
                log.write(f"Skipping {md_file}, not found.\n")
                continue
                
            report_tex = os.path.join(target_dir, f"{folder_name}_report.tex")
            report_pdf = os.path.join(target_dir, f"{folder_name}_report.pdf")
            
            with open(md_path, "r") as f:
                content = f.read()
                
            title, body = fallback_markdown_to_latex(content)
            
            with open(report_tex, "w") as f:
                f.write(LATEX_PREAMBLE % {"title": title})
                f.write(body)
                f.write(LATEX_POSTAMBLE)
                
            log.write(f"Generated {report_tex}.\n")
            
            old_tex = os.path.join(target_dir, "main.tex")
            old_pdf = os.path.join(target_dir, "main.pdf")
            old_report_pdf = os.path.join(target_dir, f"{folder_name}_report.pdf")
            if os.path.exists(old_tex): os.remove(old_tex)
            if os.path.exists(old_pdf): os.remove(old_pdf)
            # Remove previous PDF if it exists to ensure a fresh build
            if os.path.exists(old_report_pdf): os.remove(old_report_pdf)
            
            try:
                subprocess.run([TECTONIC_BIN, f"{folder_name}_report.tex"], cwd=target_dir, check=True, capture_output=True)
                log.write(f"Successfully compiled {report_pdf}\n")
            except subprocess.CalledProcessError as e:
                log.write(f"tectonic error on {md_file}: {e.stderr.decode()}\n")

if __name__ == "__main__":
    main()

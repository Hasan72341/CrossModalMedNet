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

LATEX_TEMPLATE = r"""\documentclass{article}
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
\usepackage{longtable}

\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  backgroundcolor=\color{gray!5},
  xleftmargin=10pt,
  xrightmargin=10pt,
  aboveskip=15pt,
  belowskip=15pt
}

\title{%(title)s}
\author{Documentation Bot \\ Project Group 5}
\date{}

\begin{document}
\maketitle

%(body)s

\end{document}
"""

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
            code = '\n'.join(lines[1:-1])
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
                    parsed_lines.append(r"\hline")
                if '---' in line_stripped:
                    parsed_lines.append(r"\hline")
                    continue
                row = line_stripped.strip('|').split('|')
                row_tex = []
                for cell in row:
                    cell = cell.strip().replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
                    cell = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', cell)
                    row_tex.append(cell)
                parsed_lines.append(" & ".join(row_tex) + r" \\ \hline")
                continue
            else:
                if in_table:
                    in_table = False
                    parsed_lines.append(r"\end{tabular}")
                    parsed_lines.append(r"\end{center}")

            m_h3 = re.match(r'^###\s+(.*)', line_stripped)
            if m_h3:
                parsed_lines.append(r"\subsubsection*{" + m_h3.group(1).replace('_', r'\_') + "}")
                continue
            m_h2 = re.match(r'^##\s+(.*)', line_stripped)
            if m_h2:
                parsed_lines.append(r"\subsection*{" + m_h2.group(1).replace('_', r'\_') + "}")
                continue
            m_h1 = re.match(r'^#\s+(.*)', line_stripped)
            if m_h1:
                parsed_lines.append(r"\section*{" + m_h1.group(1).replace('_', r'\_') + "}")
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
            parsed_lines.append(r"\end{tabular}")
            parsed_lines.append(r"\end{center}")
            
        tex_parts.append('\n'.join(parsed_lines))
        
    body = "\n".join(tex_parts)
    body = body.replace(r'\_\_', r'_')
    body = body.replace(r'\$F(G(CT)) \textbackslash{}approx CT\$', r'$F(G(CT)) \approx CT$')
    body = body.replace(r'\$G(F(MRI)) \textbackslash{}approx MRI\$', r'$G(F(MRI)) \approx MRI$')

    return title, body

def main():
    has_pandoc = shutil.which("pandoc") is not None
    
    with open(os.path.join(DOCS_DIR, "build_log.txt"), "w") as log:
        log.write(f"Pandoc available: {has_pandoc}\n")
        
        for md_file in FILES_TO_PROCESS:
            md_path = os.path.join(DOCS_DIR, md_file)
            if not os.path.exists(md_path):
                log.write(f"Skipping {md_file}, not found.\n")
                continue
                
            folder_name = md_file.replace("_documentation.md", "").replace(".md", "")
            target_dir = os.path.join(DOCS_DIR, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            
            target_md = os.path.join(target_dir, md_file)
            shutil.copy2(md_path, target_md)
            
            tex_file = os.path.join(target_dir, "main.tex")
            pdf_file = os.path.join(target_dir, f"{folder_name}_report.pdf")
            
            if has_pandoc:
                try:
                    with open(md_path, "r") as f:
                        content = f.read()
                    content = content.replace("```mermaid", "```text")
                    tmp_md = os.path.join(target_dir, "tmp.md")
                    with open(tmp_md, "w") as f:
                        f.write(content)
                        
                    subprocess.run([
                        "pandoc", tmp_md, "-s", "-o", tex_file,
                        "-V", "geometry:margin=1.2in",
                        "-V", "colorlinks=true"
                    ], check=True)
                    os.remove(tmp_md)
                    log.write(f"Converted {md_file} using Pandoc.\n")
                except subprocess.CalledProcessError as e:
                    log.write(f"Pandoc failed on {md_file}: {e}. Falling back to custom parser.\n")
                    has_pandoc = False
            
            if not has_pandoc:
                with open(md_path, "r") as f:
                    content = f.read()
                title, body = fallback_markdown_to_latex(content)
                tex_content = LATEX_TEMPLATE % {"title": title, "body": body}
                with open(tex_file, "w") as f:
                    f.write(tex_content)
                log.write(f"Converted {md_file} using fallback parser.\n")
            
            try:
                res1 = subprocess.run([TECTONIC_BIN, "main.tex"], cwd=target_dir, capture_output=True, text=True)
                
                # Rename main.pdf to specific name
                if os.path.exists(os.path.join(target_dir, "main.pdf")):
                    shutil.move(os.path.join(target_dir, "main.pdf"), pdf_file)
                    log.write(f"Successfully compiled {pdf_file}\n")
                else:
                    log.write(f"Failed to compile {md_file}. PDF not generated.\n")
                    log.write(f"tectonic output: {res1.stdout[:1000]}\n")
            except Exception as e:
                log.write(f"tectonic error on {md_file}: {e}\n")

if __name__ == "__main__":
    main()

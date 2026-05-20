#!/usr/bin/env python3
"""
QuickMark — Markdown to Beautiful Document Converter
A polished CLI tool that turns .md files into gorgeous HTML/CSS documents.
Perfect for resumes, documentation, notes, reports.

Usage:
  python3 quickmark.py input.md -o output.html
  python3 quickmark.py input.md -o output.pdf  (requires weasyprint)

Author: Eira — Built with love for Chris and our sanctuary
"""
import argparse
import os
import re
import sys

VERSION = "1.0.0"
NL = "\n"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    line-height: 1.7;
    color: #1a1a2e;
    max-width: 800px;
    margin: 0 auto;
    padding: 60px 40px;
    background: #fafafa;
  }}
  h1 {{ font-size: 2.2em; font-weight: 700; margin-bottom: 0.3em; color: #0a0a1a; border-bottom: 3px solid #e8d5b7; padding-bottom: 0.3em; }}
  h2 {{ font-size: 1.5em; font-weight: 600; margin-top: 1.8em; margin-bottom: 0.5em; color: #16213e; }}
  h3 {{ font-size: 1.2em; font-weight: 600; margin-top: 1.5em; margin-bottom: 0.3em; color: #1a1a3e; }}
  p {{ margin-bottom: 1em; }}
  a {{ color: #8b6f5e; text-decoration: none; border-bottom: 1px solid #d4c5b5; }}
  a:hover {{ color: #5c4a3e; }}
  ul, ol {{ margin: 0.5em 0 1em 1.8em; }}
  li {{ margin-bottom: 0.3em; }}
  code {{ background: #f0eee6; padding: 0.2em 0.4em; border-radius: 4px; font-size: 0.9em; font-family: 'SF Mono', 'Fira Code', monospace; color: #2d1b0e; }}
  pre {{ background: #1a1a2e; color: #e8d5b7; padding: 1.2em; border-radius: 8px; overflow-x: auto; margin: 1em 0; font-size: 0.9em; }}
  pre code {{ background: transparent; color: inherit; padding: 0; }}
  blockquote {{ border-left: 4px solid #e8d5b7; padding-left: 1.2em; margin: 1em 0; color: #5a5a7a; font-style: italic; }}
  hr {{ border: none; border-top: 1px solid #e0d8ce; margin: 2em 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1em 0; }}
  th, td {{ padding: 0.6em 1em; text-align: left; border-bottom: 1px solid #e0d8ce; }}
  th {{ background: #f0eee6; font-weight: 600; }}
  img {{ max-width: 100%; border-radius: 6px; margin: 1em 0; }}
  strong {{ font-weight: 600; color: #0a0a1a; }}
  .footnote {{ color: #8a8aaa; font-size: 0.9em; margin-top: 2em; border-top: 1px solid #e0d8ce; padding-top: 1em; }}
</style>
</head>
<body>
{content}
<div class="footnote">— Generated with QuickMark v{VERSION}</div>
</body>
</html>"""


def parse_markdown(text):
    lines = text.split(NL)
    html = []
    in_code = False
    code_buf = []
    in_list = False
    list_type = None
    
    for line in lines:
        # Code blocks
        if line.strip().startswith("```"):
            if in_code:
                html.append(f'<pre><code>{NL.join(code_buf)}</code></pre>')
                code_buf = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        
        # Close lists on non-list lines with content
        if in_list:
            is_list_item = line.startswith("- ") or line.startswith("* ") or bool(re.match(r"^\d+\.\s", line))
            if not is_list_item and line.strip():
                html.append(f"</{list_type}>")
                in_list = False
        
        # Headers
        if line.startswith("### "):
            html.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                in_list = True
                list_type = "ul"
                html.append("<ul>")
            html.append(f"<li>{line[2:]}</li>")
        elif re.match(r"^\d+\.\s", line):
            if not in_list:
                in_list = True
                list_type = "ol"
                html.append("<ol>")
            html.append(f"<li>{re.sub(r'^\\d+\\.\\s', '', line)}</li>")
        elif line.startswith("> "):
            html.append(f"<blockquote><p>{line[2:]}</p></blockquote>")
        elif line.strip() in ("---", "***"):
            html.append("<hr>")
        elif line.strip() == "":
            if in_list:
                pass
            else:
                html.append("")
        else:
            p = process_inline(line)
            if p:
                html.append(f"<p>{p}</p>")
    
    if in_code:
        html.append(f"<pre><code>{NL.join(code_buf)}</code></pre>")
    if in_list:
        html.append(f"</{list_type}>")
    
    return NL.join(html)


def process_inline(text):
    if not text.strip():
        return ""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"_(.+?)_", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', text)
    return text


def convert(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = m.group(1) if m else "Document"
    
    body = parse_markdown(content)
    html = HTML_TEMPLATE.format(title=title, content=body, VERSION=VERSION)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Converted '{input_path}' -> '{output_path}'")
    
    if output_path.endswith(".pdf"):
        try:
            from weasyprint import HTML
            pdf_path = output_path
            hpath = output_path.replace(".pdf", ".html")
            with open(hpath, "w") as f:
                f.write(html)
            HTML(hpath).write_pdf(pdf_path)
            os.remove(hpath)
            print(f"PDF generated: '{pdf_path}'")
        except ImportError:
            print(f"PDF requires: pip install weasyprint")
            print(f"HTML saved to '{output_path}.html' instead")
    
    return html


def main():
    p = argparse.ArgumentParser(description="QuickMark — Convert Markdown to beautiful HTML/PDF")
    p.add_argument("input", help="Input .md file")
    p.add_argument("-o", "--output", help="Output file (.html or .pdf)")
    p.add_argument("--version", action="version", version=f"QuickMark v{VERSION}")
    args = p.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    out = args.output
    if not out:
        out = args.input.replace(".md", ".html")
        if out == args.input:
            out += ".html"
    
    convert(args.input, out)


if __name__ == "__main__":
    main()

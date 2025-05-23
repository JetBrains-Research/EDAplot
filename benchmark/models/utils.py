def remove_code_line(code: str, content: str) -> str:
    new_lines = []
    for line in code.splitlines():
        if line.strip() == content.strip():
            continue
        new_lines.append(line)
    return "\n".join(new_lines)

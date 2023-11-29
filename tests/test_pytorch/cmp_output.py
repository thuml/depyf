import glob

output_files = glob.glob("depyf_output/*/__compiled_fn_*.py") + glob.glob("depyf_output/*/__transformed_code_*.py")
output_files.sort()

# ignore lock files
output_files = [x for x in output_files if not x.endswith(".lock")]

expected_files = glob.glob("tests/depyf_output/*/__compiled_fn_*.py") + glob.glob("tests/depyf_output/*/__transformed_code_*.py")
expected_files.sort()

msg = f"len(output_files)={len(output_files)}, len(expected_files)={len(expected_files)}.\n"
msg += "Unexpected files:\n"
for x in set(output_files) - set(expected_files):
    msg += x + "\n"
msg += "Missing files:\n"
for x in set(expected_files) - set(output_files):
    msg += x + "\n"

assert len(output_files) == len(expected_files), msg

for output_file, expected_file in zip(output_files, expected_files):
    if "kernel" in output_file:
        # skip kernel files, as they contain some random code paths
        continue
    with open(output_file, "r") as f:
        output_lines = []
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                output_lines.append(line.strip())
    with open(expected_file, "r") as f:
        expected_lines = []
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                expected_lines.append(line.strip())
    msg = ""
    msg += f"output_file={output_file}\n"
    msg += f"expected_file={expected_file}\n"
    msg += f"len(output_lines)={len(output_lines)}\n"
    msg += f"len(expected_lines)={len(expected_lines)}\n"
    msg += f"output_lines:\n{output_lines}\n"
    msg += f"expected_lines:\n{expected_lines}\n"
    assert len(output_lines) == len(expected_lines), msg
    # sometimes the lines are not in the same order, some lines are switched without changing the behavior of the code.
    output_lines.sort()
    expected_lines.sort()
    for output_line, expected_line in zip(output_lines, expected_lines):
        msg = ""
        msg += f"output_file={output_file}\n"
        msg += f"expected_file={expected_file}\n"
        msg += f"output_line={output_line}\n"
        msg += f"expected_line={expected_line}\n"
        assert output_line == expected_line, msg

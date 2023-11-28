import glob

output_files = glob.glob("depyf_output/*/*.py")
output_files.sort()

expected_files = glob.glob("tests/depyf_output/*/*.py")
expected_files.sort()

assert len(output_files) == len(expected_files)

for output_file, expected_file in zip(output_files, expected_files):
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
    assert output_lines == expected_lines

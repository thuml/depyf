count = 0

for line in open('tests/test_pytorch/debug_output.txt'):
    # when we use `pdb` and hit breakpoint, we will see `->` in the output
    # the first `->` is from `depyf.debug` and the second `->` is from user input
    if line.startswith('->'):
        count += 1

assert count == 2
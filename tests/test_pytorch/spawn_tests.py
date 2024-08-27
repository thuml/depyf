import multiprocessing
import subprocess
import os

def task():
    subprocess.run("coverage run --append  tests/test_pytorch/test_pytorch.py", shell=True, input="c\n", text=True)

if __name__ == '__main__':

    for TORCH_COMPILE_BACKEND in ["eager", "aot_eager", "inductor"]:
        for REQUIRES_GRAD in ["0", "1"]:
            for DYNAMIC_SHAPE in ["0", "1"]:
                for COMPILE_TYPE in ["function", "module"]:
                    for USAGE_TYPE in ["debug"]:
                        os.environ["TORCH_COMPILE_BACKEND"] = TORCH_COMPILE_BACKEND
                        os.environ["REQUIRES_GRAD"] = REQUIRES_GRAD
                        os.environ["DYNAMIC_SHAPE"] = DYNAMIC_SHAPE
                        os.environ["COMPILE_TYPE"] = COMPILE_TYPE
                        os.environ["USAGE_TYPE"] = USAGE_TYPE
                        ps = [multiprocessing.Process(target=task) for i in range(3)]
                        for p in ps:
                            p.start()
                        for p in ps:
                            p.join()
                        
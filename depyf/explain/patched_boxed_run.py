def patched_boxed_run(self, args_list):
    return self.module.forward(*args_list)
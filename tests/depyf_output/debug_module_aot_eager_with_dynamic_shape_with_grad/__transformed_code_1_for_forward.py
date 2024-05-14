def __transformed_code_1_for_forward(self, a, b):
    __temp_7 = __compiled_fn_7(a.size(0), a, b.size(0), b)
    x = __temp_7[0]
    if __temp_7[1]:
        return __resume_at_30_8(b, x)
    return __resume_at_38_9(b, x)

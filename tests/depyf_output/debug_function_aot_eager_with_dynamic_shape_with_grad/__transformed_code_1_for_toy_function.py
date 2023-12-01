def __transformed_code_1_for_toy_function(a, b):
    x_0 = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    __temp_7 = __compiled_fn_4(a.size(0), a, b.size(0), b)
    x = __temp_7[0]
    if __temp_7[1]:
        return __resume_at_30_5(b, x)
    return __resume_at_38_6(b, x)

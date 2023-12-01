def __transformed_code_0_for_toy_function(a, b):
    x_0 = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    __temp_1 = __compiled_fn_0(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)

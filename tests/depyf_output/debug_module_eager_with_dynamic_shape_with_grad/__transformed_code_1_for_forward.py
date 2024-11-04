def __transformed_code_1_for_forward(self, a, b):
    __temp_11, __temp_12 = __compiled_fn_7(__import_torch_dot__dynamo_dot_utils
        .call_size(a, 0), a, __import_torch_dot__dynamo_dot_utils.call_size(b, 
        0), b)
    x = __temp_11
    if __temp_12:
        return __resume_at_30_8(b, x)
    return __resume_at_38_9(b, x)

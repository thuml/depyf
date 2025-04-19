def __transformed_code_0_for_toy_function(a, b):
    tmp_1 = a
    tmp_2 = b
    graph_out_0 = __compiled_fn_1(a, b)
    x = graph_out_0[1]
    if graph_out_0[0]:
        return __resume_at_30_2(b, x)
    return __resume_at_38_3(b, x)

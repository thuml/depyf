def __transformed_code_0_for_torch_dynamo_resume_in_toy_function_at_5(b, x):
    a = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function

    tmp_1 = x
    tmp_2 = b
    graph_out_0 = __compiled_fn_5(x, b)
    return graph_out_0[0]

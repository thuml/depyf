def __transformed_code_1_for_torch_dynamo_resume_in_forward_at_15(b, x):
    a = None; self = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function

    tmp_1 = __import_torch_dot__dynamo_dot_utils
    tmp_2 = __import_torch_dot__dynamo_dot_utils.call_size
    tmp_3 = b
    __temp_10 = __import_torch_dot__dynamo_dot_utils.call_size(b, 0)
    tmp_4 = __temp_10
    tmp_5 = x
    graph_out_0 = __compiled_fn_11(__temp_10, tmp_3, x)
    return graph_out_0[0]

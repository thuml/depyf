def __transformed_code_1_for_forward(self, a, b):
    tmp_1 = __import_torch_dot__dynamo_dot_utils
    tmp_2 = __import_torch_dot__dynamo_dot_utils.call_size
    tmp_3 = a
    __temp_5 = __import_torch_dot__dynamo_dot_utils.call_size(a, 0)
    tmp_4 = __temp_5
    tmp_5 = b
    __temp_6 = tmp_2(b, 0)
    tmp_6 = __temp_6
    graph_out_0 = __compiled_fn_7(__temp_5, tmp_3, __temp_6, tmp_5)
    x = graph_out_0[1]
    if graph_out_0[0]:
        return __resume_at_30_8(b, x)
    return __resume_at_38_9(b, x)

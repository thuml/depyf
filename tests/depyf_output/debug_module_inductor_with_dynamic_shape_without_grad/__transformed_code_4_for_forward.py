def __transformed_code_4_for_forward(self, a, b):
    x = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function

    tmp_1 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter
    tmp_2 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter()
    tmp_3 = __import_torch_dot__dynamo_dot_utils.call_size
    tmp_4 = a
    tmp_5 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit
    __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit(tmp_2)
    graph_out_0 = __compiled_fn_3_85dbc40e_20fe_4dbc_a287_42c66c8a7173(
        __import_torch_dot__dynamo_dot_utils.call_size(a, 0), tmp_4, b)
    []
    __temp_16 = []
    __temp_16.extend([])
    [__temp_16]
    return graph_out_0[0]

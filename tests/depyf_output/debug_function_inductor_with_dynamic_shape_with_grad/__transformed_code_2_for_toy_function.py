def __transformed_code_2_for_toy_function(a, b):
    x = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function

    tmp_1 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter
    tmp_2 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter()
    tmp_3 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit
    __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit(tmp_2)
    graph_out_0 = __compiled_fn_1_311d4385_ecc3_4c17_bf93_6a899b4ece50(a, b)
    []
    __temp_7 = []
    __temp_7.extend([])
    [__temp_7]
    return graph_out_0[0]

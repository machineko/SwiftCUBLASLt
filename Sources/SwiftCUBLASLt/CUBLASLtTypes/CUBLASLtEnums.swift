public enum cublasLtMatmulTile: Int {
    case cublaslt_matmul_tile_undefined = 0 
    case cublaslt_matmul_tile_8x8 = 1 
    case cublaslt_matmul_tile_8x16 = 2 
    case cublaslt_matmul_tile_16x8 = 3 
    case cublaslt_matmul_tile_8x32 = 4 
    case cublaslt_matmul_tile_16x16 = 5 
    case cublaslt_matmul_tile_32x8 = 6 
    case cublaslt_matmul_tile_8x64 = 7 
    case cublaslt_matmul_tile_16x32 = 8 
    case cublaslt_matmul_tile_32x16 = 9 
    case cublaslt_matmul_tile_64x8 = 10 
    case cublaslt_matmul_tile_32x32 = 11 
    case cublaslt_matmul_tile_32x64 = 12 
    case cublaslt_matmul_tile_64x32 = 13 
    case cublaslt_matmul_tile_32x128 = 14 
    case cublaslt_matmul_tile_64x64 = 15 
    case cublaslt_matmul_tile_128x32 = 16 
    case cublaslt_matmul_tile_64x128 = 17 
    case cublaslt_matmul_tile_128x64 = 18 
    case cublaslt_matmul_tile_64x256 = 19 
    case cublaslt_matmul_tile_128x128 = 20 
    case cublaslt_matmul_tile_256x64 = 21 
    case cublaslt_matmul_tile_64x512 = 22 
    case cublaslt_matmul_tile_128x256 = 23 
    case cublaslt_matmul_tile_256x128 = 24 
    case cublaslt_matmul_tile_512x64 = 25 
    case cublaslt_matmul_tile_64x96 = 26 
    case cublaslt_matmul_tile_96x64 = 27 
    case cublaslt_matmul_tile_96x128 = 28 
    case cublaslt_matmul_tile_128x160 = 29 
    case cublaslt_matmul_tile_160x128 = 30 
    case cublaslt_matmul_tile_192x128 = 31 
    case cublaslt_matmul_tile_128x192 = 32 
    case cublaslt_matmul_tile_128x96 = 33 
    case cublaslt_matmul_tile_32x256 = 34 
    case cublaslt_matmul_tile_256x32 = 35 
}

public enum cublasLtMatmulStages: Int {
    case cublaslt_matmul_stages_undefined = 0 
    case cublaslt_matmul_stages_16x1 = 1 
    case cublaslt_matmul_stages_16x2 = 2 
    case cublaslt_matmul_stages_16x3 = 3 
    case cublaslt_matmul_stages_16x4 = 4 
    case cublaslt_matmul_stages_16x5 = 5 
    case cublaslt_matmul_stages_16x6 = 6 
    case cublaslt_matmul_stages_32x1 = 7 
    case cublaslt_matmul_stages_32x2 = 8 
    case cublaslt_matmul_stages_32x3 = 9 
    case cublaslt_matmul_stages_32x4 = 10 
    case cublaslt_matmul_stages_32x5 = 11 
    case cublaslt_matmul_stages_32x6 = 12 
    case cublaslt_matmul_stages_64x1 = 13 
    case cublaslt_matmul_stages_64x2 = 14 
    case cublaslt_matmul_stages_64x3 = 15 
    case cublaslt_matmul_stages_64x4 = 16 
    case cublaslt_matmul_stages_64x5 = 17 
    case cublaslt_matmul_stages_64x6 = 18 
    case cublaslt_matmul_stages_128x1 = 19 
    case cublaslt_matmul_stages_128x2 = 20 
    case cublaslt_matmul_stages_128x3 = 21 
    case cublaslt_matmul_stages_128x4 = 22 
    case cublaslt_matmul_stages_128x5 = 23 
    case cublaslt_matmul_stages_128x6 = 24 
    case cublaslt_matmul_stages_32x10 = 25 
    case cublaslt_matmul_stages_8x4 = 26 
    case cublaslt_matmul_stages_16x10 = 27 
    case cublaslt_matmul_stages_8x5 = 28 
    case cublaslt_matmul_stages_8x3 = 31 
    case cublaslt_matmul_stages_8xauto = 32 
    case cublaslt_matmul_stages_16xauto = 33 
    case cublaslt_matmul_stages_32xauto = 34 
    case cublaslt_matmul_stages_64xauto = 35 
    case cublaslt_matmul_stages_128xauto = 36 
}

public enum cublasLtClusterShape: Int {
    case cublaslt_cluster_shape_auto = 0 
    case cublaslt_cluster_shape_1x1x1 = 2 
    case cublaslt_cluster_shape_2x1x1 = 3 
    case cublaslt_cluster_shape_4x1x1 = 4 
    case cublaslt_cluster_shape_1x2x1 = 5 
    case cublaslt_cluster_shape_2x2x1 = 6 
    case cublaslt_cluster_shape_4x2x1 = 7 
    case cublaslt_cluster_shape_1x4x1 = 8 
    case cublaslt_cluster_shape_2x4x1 = 9 
    case cublaslt_cluster_shape_4x4x1 = 10 
    case cublaslt_cluster_shape_8x1x1 = 11 
    case cublaslt_cluster_shape_1x8x1 = 12 
    case cublaslt_cluster_shape_8x2x1 = 13 
    case cublaslt_cluster_shape_2x8x1 = 14 
    case cublaslt_cluster_shape_16x1x1 = 15 
    case cublaslt_cluster_shape_1x16x1 = 16 
    case cublaslt_cluster_shape_3x1x1 = 17 
    case cublaslt_cluster_shape_5x1x1 = 18 
    case cublaslt_cluster_shape_6x1x1 = 19 
    case cublaslt_cluster_shape_7x1x1 = 20 
    case cublaslt_cluster_shape_9x1x1 = 21 
    case cublaslt_cluster_shape_10x1x1 = 22 
    case cublaslt_cluster_shape_11x1x1 = 23 
    case cublaslt_cluster_shape_12x1x1 = 24 
    case cublaslt_cluster_shape_13x1x1 = 25 
    case cublaslt_cluster_shape_14x1x1 = 26 
    case cublaslt_cluster_shape_15x1x1 = 27 
    case cublaslt_cluster_shape_3x2x1 = 28 
    case cublaslt_cluster_shape_5x2x1 = 29 
    case cublaslt_cluster_shape_6x2x1 = 30 
    case cublaslt_cluster_shape_7x2x1 = 31 
    case cublaslt_cluster_shape_1x3x1 = 32 
    case cublaslt_cluster_shape_2x3x1 = 33 
    case cublaslt_cluster_shape_3x3x1 = 34 
    case cublaslt_cluster_shape_4x3x1 = 35 
    case cublaslt_cluster_shape_5x3x1 = 36 
    case cublaslt_cluster_shape_3x4x1 = 37 
    case cublaslt_cluster_shape_1x5x1 = 38 
    case cublaslt_cluster_shape_2x5x1 = 39 
    case cublaslt_cluster_shape_3x5x1 = 40 
    case cublaslt_cluster_shape_1x6x1 = 41 
    case cublaslt_cluster_shape_2x6x1 = 42 
    case cublaslt_cluster_shape_1x7x1 = 43 
    case cublaslt_cluster_shape_2x7x1 = 44 
    case cublaslt_cluster_shape_1x9x1 = 45 
    case cublaslt_cluster_shape_1x10x1 = 46 
    case cublaslt_cluster_shape_1x11x1 = 47 
    case cublaslt_cluster_shape_1x12x1 = 48 
    case cublaslt_cluster_shape_1x13x1 = 49 
    case cublaslt_cluster_shape_1x14x1 = 50 
    case cublaslt_cluster_shape_1x15x1 = 51 
}

public enum cublasLtMatmulInnerShape: Int {
    case cublaslt_matmul_inner_shape_undefined = 0 
    case cublaslt_matmul_inner_shape_mma884 = 1 
    case cublaslt_matmul_inner_shape_mma1684 = 2 
    case cublaslt_matmul_inner_shape_mma1688 = 3 
    case cublaslt_matmul_inner_shape_mma16816 = 4 
}

public enum cublasLtPointerMode: Int {
    case cublaslt_pointer_mode_host = 0 
    case cublaslt_pointer_mode_device = 1 
    case cublaslt_pointer_mode_device_vector = 2 
    case cublaslt_pointer_mode_alpha_device_vector_beta_zero = 3 
    case cublaslt_pointer_mode_alpha_device_vector_beta_host = 4 
}

public enum cublasLtPointerModeMask: Int {
    case cublaslt_pointer_mode_mask_host = 1 
    case cublaslt_pointer_mode_mask_device = 2 
    case cublaslt_pointer_mode_mask_device_vector = 4 
    case cublaslt_pointer_mode_mask_alpha_device_vector_beta_zero = 8 
    case cublaslt_pointer_mode_mask_alpha_device_vector_beta_host = 16 
}

public enum cublasLtOrder: Int {
    case cublaslt_order_col = 0 
    case cublaslt_order_row = 1 
    case cublaslt_order_col32 = 2 
    case cublaslt_order_col4_4r2_8c = 3 
    case cublaslt_order_col32_2r_4r4 = 4 
}

public enum cublasLtMatrixLayoutAttribute: Int {
    case cublaslt_matrix_layout_type = 0 
    case cublaslt_matrix_layout_order = 1 
    case cublaslt_matrix_layout_rows = 2 
    case cublaslt_matrix_layout_cols = 3 
    case cublaslt_matrix_layout_ld = 4 
    case cublaslt_matrix_layout_batch_count = 5 
    case cublaslt_matrix_layout_strided_batch_offset = 6 
    case cublaslt_matrix_layout_plane_offset = 7 
}

public enum cublasLtMatmulDescAttributes: Int {
    case cublaslt_matmul_desc_compute_type = 0 
    case cublaslt_matmul_desc_scale_type = 1 
    case cublaslt_matmul_desc_pointer_mode = 2 
    case cublaslt_matmul_desc_transa = 3 
    case cublaslt_matmul_desc_transb = 4 
    case cublaslt_matmul_desc_transc = 5 
    case cublaslt_matmul_desc_fill_mode = 6 
    case cublaslt_matmul_desc_epilogue = 7 
    case cublaslt_matmul_desc_bias_pointer = 8 
    case cublaslt_matmul_desc_bias_batch_stride = 10 
    case cublaslt_matmul_desc_epilogue_aux_pointer = 11 
    case cublaslt_matmul_desc_epilogue_aux_ld = 12 
    case cublaslt_matmul_desc_epilogue_aux_batch_stride = 13 
    case cublaslt_matmul_desc_alpha_vector_batch_stride = 14 
    case cublaslt_matmul_desc_sm_count_target = 15 
    case cublaslt_matmul_desc_a_scale_pointer = 17 
    case cublaslt_matmul_desc_b_scale_pointer = 18 
    case cublaslt_matmul_desc_c_scale_pointer = 19 
    case cublaslt_matmul_desc_d_scale_pointer = 20 
    case cublaslt_matmul_desc_amax_d_pointer = 21 
    case cublaslt_matmul_desc_epilogue_aux_data_type = 22 
    case cublaslt_matmul_desc_epilogue_aux_scale_pointer = 23 
    case cublaslt_matmul_desc_epilogue_aux_amax_pointer = 24 
    case cublaslt_matmul_desc_fast_accum = 25 
    case cublaslt_matmul_desc_bias_data_type = 26 
    case cublaslt_matmul_desc_atomic_sync_num_chunks_d_rows = 27 
    case cublaslt_matmul_desc_atomic_sync_num_chunks_d_cols = 28 
    case cublaslt_matmul_desc_atomic_sync_in_counters_pointer = 29 
    case cublaslt_matmul_desc_atomic_sync_out_counters_pointer = 30 
}

public enum cublasLtMatrixTransformDescAttributes: Int {
    case cublaslt_matrix_transform_desc_scale_type 
    case cublaslt_matrix_transform_desc_pointer_mode 
    case cublaslt_matrix_transform_desc_transa 
    case cublaslt_matrix_transform_desc_transb 
}

public enum cublasLtReductionScheme: Int {
    case cublaslt_reduction_scheme_none = 0 
    case cublaslt_reduction_scheme_inplace = 1 
    case cublaslt_reduction_scheme_compute_type = 2 
    case cublaslt_reduction_scheme_output_type = 4 
    case cublaslt_reduction_scheme_mask = 7 
}

public enum cublasLtEpilogue: Int {
    case cublaslt_epilogue_default = 1 
    case cublaslt_epilogue_relu = 2 
    case cublaslt_epilogue_relu_aux = 130 
    case cublaslt_epilogue_bias = 4 
    case cublaslt_epilogue_relu_bias = 6 
    case cublaslt_epilogue_relu_aux_bias = 134 
    case cublaslt_epilogue_drelu = 136 
    case cublaslt_epilogue_drelu_bgrad = 152 
    case cublaslt_epilogue_gelu = 32 
    case cublaslt_epilogue_gelu_aux = 160 
    case cublaslt_epilogue_gelu_bias = 36 
    case cublaslt_epilogue_gelu_aux_bias = 164 
    case cublaslt_epilogue_dgelu = 192 
    case cublaslt_epilogue_dgelu_bgrad = 208 
    case cublaslt_epilogue_bgrada = 256 
    case cublaslt_epilogue_bgradb = 512 
}

public enum cublasLtMatmulSearch: Int {
    case cublaslt_search_best_fit = 0 
    case cublaslt_search_limited_by_algo_id = 1 
    case cublaslt_search_reserved_02 = 2 
    case cublaslt_search_reserved_03 = 3 
    case cublaslt_search_reserved_04 = 4 
    case cublaslt_search_reserved_05 = 5 
}

public enum cublasLtMatmulPreferenceAttributes: Int {
    case cublaslt_matmul_pref_search_mode = 0 
    case cublaslt_matmul_pref_max_workspace_bytes = 1 
    case cublaslt_matmul_pref_reduction_scheme_mask = 3 
    case cublaslt_matmul_pref_min_alignment_a_bytes = 5 
    case cublaslt_matmul_pref_min_alignment_b_bytes = 6 
    case cublaslt_matmul_pref_min_alignment_c_bytes = 7 
    case cublaslt_matmul_pref_min_alignment_d_bytes = 8 
    case cublaslt_matmul_pref_max_waves_count = 9 
    case cublaslt_matmul_pref_impl_mask = 12 
}

public enum cublasLtMatmulAlgoCapAttributes: Int {
    case cublaslt_algo_cap_splitk_support = 0 
    case cublaslt_algo_cap_reduction_scheme_mask = 1 
    case cublaslt_algo_cap_cta_swizzling_support = 2 
    case cublaslt_algo_cap_strided_batch_support = 3 
    case cublaslt_algo_cap_out_of_place_result_support = 4 
    case cublaslt_algo_cap_uplo_support = 5 
    case cublaslt_algo_cap_tile_ids = 6 
    case cublaslt_algo_cap_custom_option_max = 7 
    case cublaslt_algo_cap_custom_memory_order = 10 
    case cublaslt_algo_cap_pointer_mode_mask = 11 
    case cublaslt_algo_cap_epilogue_mask = 12 
    case cublaslt_algo_cap_stages_ids = 13 
    case cublaslt_algo_cap_ld_negative = 14 
    case cublaslt_algo_cap_numerical_impl_flags = 15 
    case cublaslt_algo_cap_min_alignment_a_bytes = 16 
    case cublaslt_algo_cap_min_alignment_b_bytes = 17 
    case cublaslt_algo_cap_min_alignment_c_bytes = 18 
    case cublaslt_algo_cap_min_alignment_d_bytes = 19 
    case cublaslt_algo_cap_atomic_sync = 20 
}

public enum cublasLtMatmulAlgoConfigAttributes: Int {
    case cublaslt_algo_config_id = 0 
    case cublaslt_algo_config_tile_id = 1 
    case cublaslt_algo_config_splitk_num = 2 
    case cublaslt_algo_config_reduction_scheme = 3 
    case cublaslt_algo_config_cta_swizzling = 4 
    case cublaslt_algo_config_custom_option = 5 
    case cublaslt_algo_config_stages_id = 6 
    case cublaslt_algo_config_inner_shape_id = 7 
    case cublaslt_algo_config_cluster_shape_id = 8 
}

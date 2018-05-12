void encode_rows_cpu(THFloatTensor* input, THIntTensor* output);
void encode_cols_cpu(THFloatTensor* input, THIntTensor* output);
void binary_gemm_cpu(THIntTensor* a, THIntTensor* b, THFloatTensor* c, int m, int nn, int k, int transb, int beta, int alpha, THFloatTensor* alphas);
void THNN_Bin_SpatialConvolutionMM_updateOutput(
          THFloatTensor *input,
          THFloatTensor *output,
          THIntTensor *weight,
          THFloatTensor *bias,
          THFloatTensor *columns,
          THFloatTensor *alphas,
          int kH, int kW,
          int dH, int dW,
          int padH, int padW);
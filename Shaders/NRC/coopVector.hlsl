#pragma once

// the sizes of the vectors ( the number of components , they are also the sizes of the matrices )
#define NRC_IN_SIZE 16        // 8 numbers of the hash grid + the normal ( 3 ) , the rest is zeros
#define NRC_HIDDEN_SIZE 64
#define NRC_OUT_SIZE 16       // 3 colours + zeros : which sizes of the output the GPU takes is to be checked together with the buffers

// the type of the numbers inside a matrix / a vector ( VkComponentTypeKHR )
static const int NRC_F16 = 0;
// the layout of a matrix in memory ( VkCooperativeVectorMatrixLayoutNV ) : the optimal ones are made by vkCmdConvertCooperativeVectorMatrixNV
static const int NRC_LAYOUT_ROW_MAJOR = 0;
static const int NRC_LAYOUT_COLUMN_MAJOR = 1;
static const int NRC_LAYOUT_INFERENCING_OPTIMAL = 2;
static const int NRC_LAYOUT_TRAINING_OPTIMAL = 3;

// the offsets in BYTES inside the weight buffers : the same numbers as W1_OFFSET , B1_OFFSET , W2_OFFSET , B2_OFFSET of the NRCManager ( keep them equal )
static const int NRC_W1_OFFSET = 0;
static const int NRC_B1_OFFSET = 2048;
static const int NRC_W2_OFFSET = 0;
static const int NRC_B2_OFFSET = 2048;

// the buffers are given to the shader as ADDRESSES ( a uniform buffer with two numbers ) : the matrix operations take a pointer and an offset in BYTES from it
struct NrcBlob { uint data[1 << 20]; };
typedef vk::BufferPointer<NrcBlob , 64> NrcPtr;
// the same four addresses as NRC::Pointers of shaderStructure.h ( the C++ side fills them from the buffers of the NRCManager )
struct NrcPointers
{
    NrcPtr hiddenWeights;   // neuronHiddenBuffer : W1 and b1 ( float16_t ) , the shader only READS it
    NrcPtr outputWeights;   // neuronPerChannelBuffer : W2 and b2
    NrcPtr hiddenGrads;     // the gradients of W1 and b1 : the shader ADDS into it ( outer product , reduce sum )
    NrcPtr outputGrads;     // the gradients of W2 and b2
};

// the vectors : OpTypeCooperativeVectorNV ( the size in bytes , the alignment , the type of a component , the number of components )
typedef vk::SpirvType<5288 , NRC_IN_SIZE * 2 , 2 , float16_t , vk::integral_constant<int , NRC_IN_SIZE> > CoopIn;
typedef vk::SpirvType<5288 , NRC_HIDDEN_SIZE * 2 , 2 , float16_t , vk::integral_constant<int , NRC_HIDDEN_SIZE> > CoopHidden;
typedef vk::SpirvType<5288 , NRC_OUT_SIZE * 2 , 2 , float16_t , vk::integral_constant<int , NRC_OUT_SIZE> > CoopOut;


// layer 1 :  z = W1 * x + b1     ( W1 is NRC_HIDDEN_SIZE x NRC_IN_SIZE )
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5292)]]
CoopHidden mulAddInHidden(CoopIn input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , [[vk::ext_reference]] uint bias[1 << 20] , int biasOffset , int biasInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);

// layer 2 :  o = W2 * h + b2     ( W2 is NRC_OUT_SIZE x NRC_HIDDEN_SIZE )
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5292)]]
CoopOut mulAddHiddenOut(CoopHidden input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , [[vk::ext_reference]] uint bias[1 << 20] , int biasOffset , int biasInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);

// the signal back through layer 2 :  dh = W2^T * dOut   ( OpCooperativeVectorMatrixMulNV , no bias )
// with transpose = true M and K are the sizes of the operation AFTER the transposing : M = NRC_HIDDEN_SIZE ( the size of the result ) , K = NRC_OUT_SIZE ( the size of the input ) ,
// the matrix itself lies in memory as it is ( NRC_OUT_SIZE rows ) , the stride is the bytes of ITS row
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5289)]]
CoopHidden mulTransposeOutHidden(CoopOut input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);

// the signal back through layer 1 :  dx = W1^T * dz   ( the signal for the numbers that came from the hash grid ) , M = NRC_IN_SIZE , K = NRC_HIDDEN_SIZE
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5289)]]
CoopIn mulTransposeHiddenIn(CoopHidden input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);

// ---------------------------------------------------------------- the gradients ( CooperativeVectorTrainingNV )
// the gradient of a matrix :  matrix[ i ][ j ] += a[ i ] * b[ j ]   ( OpCooperativeVectorOuterProductAccumulateNV ) . every thread gives its own a and b , the extension adds the products
// of ALL the threads of the wave into the buffer itself : no atomic add of your own . a is the signal that comes from above ( dOut or dz ) , b is what went through the wheel ( h or x )
[[vk::ext_capability(5435)]]
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5290)]]
void outerProductOutHidden([[vk::ext_reference]] uint pointer[1 << 20] , int offset , CoopOut a , CoopHidden b , int layout , int matrixInterp , uint stride);

[[vk::ext_capability(5435)]]
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5290)]]
void outerProductHiddenIn([[vk::ext_reference]] uint pointer[1 << 20] , int offset , CoopHidden a , CoopIn b , int layout , int matrixInterp , uint stride);

// the gradient of a bias :  bias[ i ] += v[ i ]   ( OpCooperativeVectorReduceSumAccumulateNV ) , the vectors of the threads of the wave are summed into the buffer
[[vk::ext_capability(5435)]]
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5291)]]
void reduceSumOut([[vk::ext_reference]] uint pointer[1 << 20] , int offset , CoopOut v);

[[vk::ext_capability(5435)]]
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5291)]]
void reduceSumHidden([[vk::ext_reference]] uint pointer[1 << 20] , int offset , CoopHidden v);

// ---------------------------------------------------------------- the vectors : to build , to read , to count
// OpCompositeConstruct needs a number for EVERY component ( one number for all of them is not taken ) : the input is built from its 16 numbers
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
CoopIn constructIn(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 ,
                   float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15);

// the same for the output vector ( 16 numbers : the first 3 are the real channels , the rest are zeros )
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
CoopOut constructOut(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 ,
                     float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15);

// one component with the number written in the code ( OpCompositeExtract ) : the index is a literal , it cannot be a variable
[[vk::ext_instruction(81)]]
float16_t extractOut(CoopOut v , [[vk::ext_literal]] int index);

// the same for the input vector ( the signal dx that comes back to the hash grid : the first 8 components )
[[vk::ext_instruction(81)]]
float16_t extractIn(CoopIn v , [[vk::ext_literal]] int index);

// one component with an index that is computed ( OpVectorExtractDynamic ) and to change one component ( OpVectorInsertDynamic )
[[vk::ext_instruction(77)]]
float16_t extractOutDynamic(CoopOut v , int index);
[[vk::ext_instruction(77)]]
float16_t extractHiddenDynamic(CoopHidden v , int index);
[[vk::ext_instruction(77)]]
float16_t extractInDynamic(CoopIn v , int index);
[[vk::ext_instruction(78)]]
CoopIn insertInDynamic(CoopIn v , float16_t component , int index);
[[vk::ext_instruction(78)]]
CoopOut insertOutDynamic(CoopOut v , float16_t component , int index);

// the arithmetic of two vectors of the same size , component by component
[[vk::ext_instruction(129)]]   // OpFAdd
CoopOut addOut(CoopOut a , CoopOut b);
[[vk::ext_instruction(131)]]   // OpFSub
CoopOut subOut(CoopOut a , CoopOut b);
[[vk::ext_instruction(131)]]   // OpFSub
CoopHidden subHidden(CoopHidden a , CoopHidden b);
[[vk::ext_instruction(133)]]   // OpFMul
CoopHidden mulHidden(CoopHidden a , CoopHidden b);
[[vk::ext_instruction(142)]]   // OpVectorTimesScalar
CoopHidden timesScalarHidden(CoopHidden a , float16_t s);
[[vk::ext_instruction(142)]]   // OpVectorTimesScalar
CoopOut timesScalarOut(CoopOut a , float16_t s);
[[vk::ext_instruction(40 , "GLSL.std.450")]]   // FMax
CoopHidden maxHidden(CoopHidden a , CoopHidden b);
[[vk::ext_instruction(48 , "GLSL.std.450")]]   // Step : 0 where x < edge , 1 where x >= edge
CoopHidden stepHidden(CoopHidden edge , CoopHidden x);

// ReLU of a vector : max( z , 0 ) . there is no way to write "0" for a whole vector , so it is z - z
CoopHidden reluHidden(CoopHidden z)
{
    const CoopHidden zero = subHidden(z , z);
    return maxHidden(z , zero);
}

// the gate of ReLU for the signal back : 1 where z was above 0 , 0 where the neuron was shut ( dz = dh * gate )
CoopHidden reluGateHidden(CoopHidden z)
{
    const CoopHidden zero = subHidden(z , z);
    return stepHidden(zero , z);
}

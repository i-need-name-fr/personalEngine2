#pragma once

// the question to the net : NRC::askNRC( P , N ) gives the light that ARRIVES at the point P ( without the albedo of the point ) .
// one file for all the shaders that ask the net ( the training shader , the ray tracing ) : the descriptors of the NRCManager and the forward pass of the net
//
// the number of the descriptor set is different in every pipeline , so the shader that includes this file says it FIRST :
//      #define NRC_SET 3
//      #include "../NRC/nrcQuery.hlsl"
// ( the layout of the set is the one of NRCManager : 9 bindings , 0 .. 7 the storage buffers , 8 the uniform buffer with the addresses )
#ifndef NRC_SET
#error "define NRC_SET ( the number of the descriptor set of the NRCManager ) before the include of nrcQuery.hlsl"
#endif

[[vk::binding(0 , NRC_SET)]] RWStructuredBuffer<NRC::_TrainingSample> trainingBuffer;
[[vk::binding(1 , NRC_SET)]] RWStructuredBuffer<float> hashGridBuffer;
[[vk::binding(2 , NRC_SET)]] RWStructuredBuffer<NRC::Parameter> hashGridParamsBuffer;
[[vk::binding(3 , NRC_SET)]] RWStructuredBuffer<NRC::HashGridInfo> hashGridGeneralBuffer;
[[vk::binding(4 , NRC_SET)]] RWStructuredBuffer<NRC::Parameter> neuronParamsHiddenBuffer;
[[vk::binding(5 , NRC_SET)]] RWStructuredBuffer<NRC::Parameter> neuronParamsPerChannelBuffer;
[[vk::binding(6 , NRC_SET)]] RWStructuredBuffer<float> neuronHiddenBuffer;
[[vk::binding(7 , NRC_SET)]] RWStructuredBuffer<float> neuronPerChannelBuffer;

// the cooperative vector operations of the net ( the matrices are multiplied on the tensor cores )
#include "coopVector.hlsl"
[[vk::binding(8 , NRC_SET)]] ConstantBuffer<NrcPointers> nrcPtr;
// one NRC::GeneralData : the size of the training grid ( width , height ) and the counter of the valid samples ( InterlockedAdd on validCount in the trace pass )
[[vk::binding(9 , NRC_SET)]] RWStructuredBuffer<NRC::GeneralData> nrcGeneral;

// float += float from many threads at once ( OpAtomicFAddEXT ) : HLSL has no InterlockedAdd for float . scope 1 = Device , semantics 0 = relaxed
// atomicAddFloat(hashGridParamsBuffer[i].grad , 1 , 0 , value)
[[vk::ext_capability(6033)]]
[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_instruction(6035)]]
float atomicAddFloat([[vk::ext_reference]] float pointer , uint scope , uint semantics , float value);

namespace NRC
{
    float softplus(float x)
    {
        return log(1.0f + exp(x));
    }


    float3 forwardNRC(const float3 P , const float3 N , out CoopIn x , out CoopHidden h, out CoopHidden z1)
    {

        NRC::HashGridResult hresult;

        [unroll(8)]
        for(uint i = 0 ; i < 8 ; i++)
        {
            hresult.data[i] = 0.0f;
        }


        [unroll]
        for(uint level = 0 ; level < MOX_NRC_MAX_HASH_GRID_SIZE ; level++)
        {

            const NRC::HashGridInfo levelData = hashGridGeneralBuffer[level];
            const float size = levelData.cellSize;
            const int T = int(levelData.T);
            const int offset = int(levelData.offset);

            const float3 q = P / size;
            int3 base = int3(floor(q));
            const float3 f = q - float3(base);
            int3 id = base % T;
            if(id.x < 0) id.x += T;
            if(id.y < 0) id.y += T;
            if(id.z < 0) id.z += T;

            float totalW1 = 0.0f;
            float totalW2 = 0.0f;

            [unroll]
            for(int z = 0 ; z < 2 ; z++)
            [unroll]
            for(int y = 0 ; y < 2 ; y++)
            [unroll]
            for(int x = 0 ; x < 2 ; x++)
            {
                const int3 index = (id + int3(x , y , z)) % T;
                const int3 angle = int3(x , y , z);
                const float3 weights = lerp(1.0f - f , f , float3(angle));
                // z = 1
                // y = 0
                // x = 0
                // 4 spot - 5 spot
                int hashIndex = index.z * T * T + index.y * T + index.x;
                hashIndex *= 2;
                hashIndex += offset;

                // since our F = 2
                const float w1 = hashGridBuffer[hashIndex];
                const float w2 = hashGridBuffer[hashIndex + 1];

                const float sumWeight = weights.x * weights.y * weights.z;

                totalW1 += sumWeight * w1;
                totalW2 += sumWeight * w2;

            }

            hresult.data[level * 2] = totalW1;
            hresult.data[level * 2 + 1] = totalW2;

        }

        const float3 inputN = N * 0.5 + 0.5;

        CoopIn input = constructIn(float16_t(hresult.data[0]) , float16_t(hresult.data[1]) , float16_t(hresult.data[2]),
                                    float16_t(hresult.data[3]) , float16_t(hresult.data[4]) , float16_t(hresult.data[5])
                                    ,float16_t(hresult.data[6]) ,float16_t(hresult.data[7]) , float16_t(inputN.x) ,
                                    float16_t(inputN.y) , float16_t(inputN.z), 0.0f , 0.0f , 0.0f ,0.0f , 0.0f);
        // CoopHidden mulAddInHidden(CoopIn input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , [[vk::ext_reference]] uint bias[1 << 20] ,
        //  int biasOffset , int biasInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);
        x = input;
        CoopHidden hiddenOutput = mulAddInHidden(input , NRC_F16 ,
                                                 nrcPtr.hiddenWeights.Get().data , NRC_W1_OFFSET , NRC_F16 ,
                                                 nrcPtr.hiddenWeights.Get().data , NRC_B1_OFFSET , NRC_F16 ,
                                                 NRC_HIDDEN_SIZE , NRC_IN_SIZE , NRC_LAYOUT_INFERENCING_OPTIMAL , false , NRC_IN_SIZE * 2 , 0);

        z1 = hiddenOutput;
        hiddenOutput = reluHidden(hiddenOutput);
        h = hiddenOutput;
        CoopOut outputVector = mulAddHiddenOut(hiddenOutput , NRC_F16 , nrcPtr.outputWeights.Get().data , NRC_W2_OFFSET , NRC_F16 ,
                                                                        nrcPtr.outputWeights.Get().data , NRC_B2_OFFSET , NRC_F16,
                                                                        NRC_OUT_SIZE , NRC_HIDDEN_SIZE , NRC_LAYOUT_INFERENCING_OPTIMAL , false ,  NRC_HIDDEN_SIZE * 2 , 0);

        const float r = float(extractOut(outputVector , 0));
        const float g = float(extractOut(outputVector , 1));
        const float b = float(extractOut(outputVector , 2));

        const float3 result = float3(r , g ,b);

        return result;
    }

    float3 askNRC(const float3 P , const float3 N)
    {
        CoopIn x;
        CoopHidden h;
        CoopHidden z1;
        const float3 answer = forwardNRC(P , N , x , h , z1);

        const float3 result = float3(softplus(answer.x) , softplus(answer.y) ,softplus(answer.z));

        return result;
    }
}

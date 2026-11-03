#include "../../include/ShaderStructures/ShaderStructure.h"

#define NRC_SET 3
#include "nrcQuery.hlsl"

float derivativeSoftplus(const float x)
{
    return 1.0f /(1.0f + exp(-x));
}

void addHashGrid(const float3 P ,const CoopIn data)
{
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
                const float sumWeight = weights.x * weights.y * weights.z;

                // since our F = 2
                atomicAddFloat(hashGridParamsBuffer[hashIndex].grad , 1 , 0 , float(extractInDynamic(data , level * 2)) * sumWeight);
                atomicAddFloat(hashGridParamsBuffer[hashIndex + 1].grad , 1 , 0 , float(extractInDynamic(data , level * 2 + 1)) * sumWeight);
            }
        }
}

// EVERY thread of the wave must come here ( outerProduct and reduceSum add the vectors of the whole wave ) : a thread without a valid sample gives the zero signal
void processUpdate(const NRC::_TrainingSample input , const bool valid)
{
    // first of all we have to ask a question to NRC 
    // to see the result

    CoopIn x;
    CoopHidden h;
    CoopHidden z1;
    const float3 _nrcAnswer = NRC::forwardNRC(input.position , input.normal , x,  h , z1);
    const float3 NRC_result = float3(NRC::softplus(_nrcAnswer.x) , NRC::softplus(_nrcAnswer.y) , NRC::softplus(_nrcAnswer.z));

    // now we find the dL/dout
    const float3 dL_dout = 2.0f * (NRC_result - input.radiance) * (valid ? 1.0f : 0.0f);
    CoopOut firstLayerInput = constructOut(float16_t(dL_dout.x) * float16_t(derivativeSoftplus(_nrcAnswer.x)), float16_t(dL_dout.y) * float16_t(derivativeSoftplus(_nrcAnswer.y)) , float16_t(dL_dout.z) *float16_t(derivativeSoftplus(_nrcAnswer.z)) , 0.0f , 0.0f , 0.0f , 0.0 , 0.0, 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0);
    outerProductOutHidden(nrcPtr.outputGrads.Get().data , 0 , firstLayerInput , h , NRC_LAYOUT_TRAINING_OPTIMAL , NRC_F16 , NRC_HIDDEN_SIZE * 2);
    CoopHidden dL_dh = mulTransposeOutHidden(firstLayerInput , NRC_F16 , nrcPtr.outputWeights.Get().data , NRC_W2_TRAIN_OFFSET , NRC_F16 ,
                                            NRC_HIDDEN_SIZE , NRC_OUT_SIZE, NRC_LAYOUT_TRAINING_OPTIMAL, // we change the M and K since we matrix is transposed
                                            true , NRC_HIDDEN_SIZE * 2 , 0);
    CoopHidden gate = reluGateHidden(z1);
    CoopHidden dz1 = mulHidden(dL_dh , gate);

    CoopIn inputCoop = mulTransposeHiddenIn(dz1 , NRC_F16 , nrcPtr.hiddenWeights.Get().data , NRC_W1_TRAIN_OFFSET , NRC_F16 ,
                                            NRC_IN_SIZE , NRC_HIDDEN_SIZE, NRC_LAYOUT_TRAINING_OPTIMAL, // we change the M and K since we matrix is transposed
                                            true , NRC_IN_SIZE * 2 , 0);

    outerProductHiddenIn(nrcPtr.hiddenGrads.Get().data , 0 , dz1 , x , NRC_LAYOUT_TRAINING_OPTIMAL , NRC_F16 , NRC_IN_SIZE * 2);
    reduceSumHidden(nrcPtr.hiddenGrads.Get().data , NRC_B1_GRAD_OFFSET , dz1);
    reduceSumOut(nrcPtr.outputGrads.Get().data , NRC_B2_GRAD_OFFSET , firstLayerInput);
    // now we update the hash grid
    // the atomics are not cooperative : the invalid threads can skip it
    if(valid) addHashGrid(input.position , inputCoop);
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    // first we get our 3 samples
    const uint2 size = uint2(nrcGeneral[0].width , nrcGeneral[0].height);
    // no return : the threads over the edge ( the dispatch is rounded up ) also have to take part in the cooperative operations , they just have no valid sample
    const bool inside = all(id.xy < size);
    uint bufferID = bufferIndex(min(id.xy , size - 1) , size.x , size.y);
    bufferID *= MOX_NRC_MAX_BOUNCE_COUNT;
    NRC::_TrainingSample samples[MOX_NRC_MAX_BOUNCE_COUNT];


    [unroll]
    for(uint i = 0 ; i < MOX_NRC_MAX_BOUNCE_COUNT ; i++)
    {
        samples[i] = trainingBuffer[bufferID + i];
        const bool valid = inside && samples[i].valid == MOX_NRC_VALID_SAMPLE;
        if(valid)
        {
            InterlockedAdd(nrcGeneral[0].validCount , 1);
        }
        processUpdate(samples[i] , valid);
    }
}
#include "../../include/ShaderStructures/ShaderStructure.h"

#define NRC_SET 3
#include "nrcQuery.hlsl"

#define ALPHA_TEST_SAMPLER typicalSampler

#define NRC_LAYER_HIDDEN 0
#define NRC_LAYER_OUT 1
#define NRC_HASH_GRID_START 2

#define NRC_HASH_GRID_SIZE 4

float readHalf(const NrcPtr buffer , const uint halfIndex)
{
    const uint word = buffer.Get().data[halfIndex >> 1];
    return f16tof32(word >> ((halfIndex & 1) * 16));
}

void writeHalf(const NrcPtr buffer , const uint halfIndex , const float value)
{
    const uint shift = (halfIndex & 1) * 16;
    InterlockedAnd(buffer.Get().data[halfIndex >> 1] , ~(0xFFFFu << shift));
    InterlockedOr(buffer.Get().data[halfIndex >> 1] , f32tof16(value) << shift);
}

void adamStep(inout NRC::Parameter p , const float g)
{
    static const float b1 = 0.9f;
    static const float b2 = 0.999f;
    p.m = p.m * b1 + (1.0f - b1) * g;
    p.v = p.v * b2 + (1.0f - b2) * g * g;
    const float m = p.m / (1.0f - pow(b1 , float(nrcGeneral[0].stepCount)));
    const float v = p.v / (1.0f - pow(b2 , float(nrcGeneral[0].stepCount)));
    const float epsilon = 1e-8f;
    p.w -= 1e-3f * m / (sqrt(v) + epsilon);
    
}

void recordAdam_hidden(const uint3 id) 
{
    const uint validCount = nrcGeneral[0].validCount;
    if(validCount == 0) return;
    const uint pos = id.x;
    const uint totalSize = NRC_HIDDEN_SIZE * NRC_IN_SIZE + NRC_HIDDEN_SIZE; //
    if(pos >= totalSize) return;
    float g = 0.0f;

    if(pos < NRC_HIDDEN_SIZE * NRC_OUT_SIZE)
    {
        g = gradStagingHidden[pos] / (float(validCount) * NRC_GRAD_SCALE);
    }
    else
    {
        // the bias : its gradient is fp16 inside hiddenGrads after the TRAINING_OPTIMAL matrix
        const uint b = pos - NRC_HIDDEN_SIZE * NRC_IN_SIZE;
        g = readHalf(nrcPtr.hiddenGrads , NRC_B1_GRAD_OFFSET / 2 + b) / (float(validCount) * NRC_GRAD_SCALE);
    }



    NRC::Parameter p = neuronParamsHiddenBuffer[pos];
    if(!isfinite(g)) return;
    adamStep(p , g);

    neuronParamsHiddenBuffer[pos] = p;
    if(pos < NRC_HIDDEN_SIZE * NRC_IN_SIZE)
    {
        weightStagingHidden[pos] = p.w;
    }
    else
    {
        writeHalf(nrcPtr.hiddenWeights , NRC_B1_OFFSET / 2 + (pos - NRC_HIDDEN_SIZE * NRC_IN_SIZE) , p.w);
    }
}

void recordAdam_output(const uint3 id)
{
    const uint validCount = nrcGeneral[0].validCount;
    if(validCount == 0) return;
    const uint pos = id.x;
    const uint totalSize = NRC_HIDDEN_SIZE * NRC_OUT_SIZE + NRC_HIDDEN_SIZE; //
    if(pos >= totalSize) return;
    float g = 0.0f;

    if(pos < NRC_HIDDEN_SIZE * NRC_OUT_SIZE)
    {
        g = gradStagingOutput[pos] / (float(validCount) * NRC_GRAD_SCALE);
    }
    else
    {
        const uint b = pos - NRC_HIDDEN_SIZE * NRC_OUT_SIZE;
        g = readHalf(nrcPtr.outputGrads , NRC_B2_GRAD_OFFSET / 2 + b) / (float(validCount) * NRC_GRAD_SCALE);
    }



    NRC::Parameter p = neuronParamsPerChannelBuffer[pos];
    if(!isfinite(g)) return;

    adamStep(p , g);

    neuronParamsPerChannelBuffer[pos] = p;
    if(pos < NRC_HIDDEN_SIZE * NRC_OUT_SIZE)
    {
        weightStagingOutput[pos] = p.w;
    }
    else
    {
        // the bias has no layout , it goes straight into the weight buffer
        writeHalf(nrcPtr.outputWeights , NRC_B2_OFFSET / 2 + (pos - NRC_HIDDEN_SIZE * NRC_OUT_SIZE) , p.w);
    }
}

void recordAdam_hash(const uint3 id)
{
    if(id.y < NRC_HASH_GRID_START)  return;
    const uint validCount = nrcGeneral[0].validCount;
    const uint layer = id.y - NRC_HASH_GRID_START;
    const NRC::HashGridInfo levelData = hashGridGeneralBuffer[layer];
    const int T = int(levelData.T);
    const int offset = int(levelData.offset);
    if(validCount == 0) return;
    uint pos = id.x;
    const uint totalSize = T * T * T * 2; //
    if(pos >= totalSize) return;
    pos += offset;
    NRC::Parameter p = hashGridParamsBuffer[pos];
    const float g = p.grad / (float(validCount) * NRC_GRAD_SCALE);
    if(!isfinite(g))
    {
        // the sum is in the memory for the next step too : a NaN would stay there for ever
        hashGridParamsBuffer[pos].grad = 0.0f;
        return;
    }

    adamStep(p , g);
    p.grad = 0;
    // we dont care about float16_t style here , since its our personal hash grid
    hashGridParamsBuffer[pos] = p;
    hashGridBuffer[pos] = p.w;
}


[shader("compute")]
[numthreads(128 , 1 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    switch(id.y)
    {
        case NRC_LAYER_HIDDEN: recordAdam_hidden(id); break;
        case NRC_LAYER_OUT:    recordAdam_output(id); break;
        default:               recordAdam_hash(id);   break;
    }
}
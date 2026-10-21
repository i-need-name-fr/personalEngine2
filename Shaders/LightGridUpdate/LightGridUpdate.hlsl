#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] RWStructuredBuffer<lg::LightSource> lightBuffer;
[[vk::binding(1 , 0)]] RWStructuredBuffer<lg::LightCell> lightCells;
[[vk::binding(2 , 0)]] ConstantBuffer<lg::GridCascadeInfo> gridInfo;

[[vk::push_constant]] struct {
    float3 origin;
    uint mode;
} push;

uint gridCells(){
    return (uint)lg::GRID_SIZE;
}

[shader("compute")]
[numthreads(256 , 1 , 1)]
void main(uint3 id : SV_DispatchThreadID){
    const uint n = gridCells();

    if(push.mode == 0u){
        const uint totalCells = n * n * n;
        if(id.x >= totalCells) return;

        const uint x = id.x % n;
        const uint y = (id.x / n) % n;
        const uint z = id.x / (n * n);

        lightCells[id.x].position = push.origin + (float3(x , y , z) + 0.5f) * lg::GRID_CELL_SIZE;
        lightCells[id.x].totalLightCount = 0u;
        return;
    }

    if(id.x >= lg::MAX_LIGHT_SOURCES) return;

    const lg::LightSource light = lightBuffer[id.x];
    if(light.radius <= 0.0f) return;

    const float3 gridPos = light.position - push.origin;
    const int3 low = (int3)floor((gridPos - light.radius) / lg::GRID_CELL_SIZE);
    const int3 high = (int3)floor((gridPos + light.radius) / lg::GRID_CELL_SIZE);

    const int count = (int)n;
    if(any(high < int3(0 , 0 , 0)) || any(low >= int3(count , count , count))) return;

    const int3 first = max(low , int3(0 , 0 , 0));
    const int3 last = min(high , int3(count - 1 , count - 1 , count - 1));

    for(int z = first.z ; z <= last.z ; z++){
        for(int y = first.y ; y <= last.y ; y++){
            for(int x = first.x ; x <= last.x ; x++){
                const float3 cellMin = float3(x , y , z) * lg::GRID_CELL_SIZE;
                const float3 cellMax = cellMin + lg::GRID_CELL_SIZE;
                const float3 toLight = gridPos - clamp(gridPos , cellMin , cellMax);
                if(dot(toLight , toLight) > light.radius * light.radius) continue;

                const uint cellIndex = (uint)x + n * ((uint)y + n * (uint)z);

                uint slot;
                InterlockedAdd(lightCells[cellIndex].totalLightCount , 1u , slot);
                if(slot < lg::MAX_LIGHT_PER_CELL){
                    lightCells[cellIndex].lightIndices[slot] = id.x;
                }
            }
        }
    }
}

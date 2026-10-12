#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::image_format("r32f")]]
[[vk::binding(0 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("rgba32f")]]
[[vk::binding(1 , 0)]] RWTexture2D<float4> position;

[[vk::image_format("rgba16f")]]
[[vk::binding(2 , 0)]] RWTexture2D<float4> normal;

[[vk::image_format("r32f")]]
[[vk::binding(3 , 0)]] RWTexture2D<float> history;

[[vk::image_format("r32f")]]
[[vk::binding(4 , 0)]] RWTexture2D<float> output;

[[vk::binding(5 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(6 , 0)]] ConstantBuffer<GeneralFeatures> features;

[[vk::binding(7 , 0)]] RWStructuredBuffer<float4> SSAO_Kernel;

// a fixed number between 0 and 1 for every pixel ( interleaved gradient noise ) : it turns the kernel / the directions of a pixel
// a bit differently from its neighbours , and it is the same in every frame , so nothing flickers . the bilateral filter after this
// pass averages the pattern out
float pixelDither(uint2 pixel){
    return frac(52.9829189f * frac(dot(float2(pixel) , float2(0.06711056f , 0.00583715f))));
}

bool isSkyPixel(uint2 coord){
    // the G-buffer is cleared to 0 where nothing was drawn
    const float3 n = normal[coord].xyz;
    return dot(n , n) < 0.5f;
}

// the normal of the GEOMETRY , from the positions of the neighbouring pixels . the normal of the G-buffer has the normal map in it : the
// bumps of the stone tilt it by up to 40 degrees , and a flat floor looked like it was covered by its own bricks ( the AO showed the
// pattern of the texture ) . the closer neighbour is taken on each axis , so an edge of an object does not bend the normal
float3 geometricNormal(uint2 coord , float3 P , float3 fallback){
    const uint2 maxCoord = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight) - 1u;

    const uint2 right = uint2(min(coord.x + 1u , maxCoord.x) , coord.y);
    const uint2 left = uint2(coord.x - (coord.x > 0u ? 1u : 0u) , coord.y);
    const uint2 down = uint2(coord.x , min(coord.y + 1u , maxCoord.y));
    const uint2 up = uint2(coord.x , coord.y - (coord.y > 0u ? 1u : 0u));

    const float3 toRight = position[right].xyz - P;
    const float3 toLeft = P - position[left].xyz;
    const float3 toDown = position[down].xyz - P;
    const float3 toUp = P - position[up].xyz;

    // a sky pixel has a position of 0 , far from everything : its neighbour is never the closer one
    const float3 dx = (dot(toRight , toRight) < dot(toLeft , toLeft)) ? toRight : toLeft;
    const float3 dy = (dot(toDown , toDown) < dot(toUp , toUp)) ? toDown : toUp;

    // the screen y goes down , so the order of the cross product turns the normal to the camera
    float3 n = cross(dy , dx);
    const float len = length(n);
    if(len < 1e-8f) return fallback;
    n /= len;

    // it has to face the camera ( the sign above depends on which neighbours were taken )
    const float3 toCamera = generalBuffer.playerPosition - P;
    if(dot(n , toCamera) < 0.0f) n = -n;

    // a neighbour of another surface gives a wrong normal : then the one of the G-buffer is the better guess
    return (dot(n , fallback) > 0.5f) ? n : fallback;
}

// SSAO in the view space ( the camera looks down -z ) :
//   -- the old version took the VIEW DIRECTION as the position of the pixel , so every sample was taken around the camera ;
//   -- it compared the linear depth ( positive ) with a view space z ( negative ) ;
//   -- it did not flip y when going from the clip space to the pixel , the engine does ( see the viewport )
float computeSSAO(uint2 currentPixel , float3 viewPos , float3 viewNormal , float radius , uint iteration){
    const uint iterations = max(min(iteration , 64u) , 1u);
    const uint2 screenSize = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);

    // the kernel is turned around the normal by an angle that depends on the pixel
    const float angle = pixelDither(currentPixel) * 2.0f * PI;
    float3 rvec = float3(cos(angle) , sin(angle) , 0.0f);
    float3 projected = rvec - viewNormal * dot(viewNormal , rvec);
    if(dot(projected , projected) < 1e-6f){
        rvec = float3(0.0f , 1.0f , 0.0f);
        projected = rvec - viewNormal * dot(viewNormal , rvec);
    }
    const float3 tangent = normalize(projected);
    const float3 bitangent = cross(viewNormal , tangent);
    const float3x3 TBN = float3x3(tangent , bitangent , viewNormal);

    float occlusion = 0.0f;
    for(uint i = 0 ; i < iterations ; i++){
        // tangent space -> view space : the vector on the left , the rows of TBN are T , B , N
        const float3 offset = mul(SSAO_Kernel[i].xyz , TBN);
        const float3 samplePos = viewPos + offset * radius;

        const float4 clip = mul(generalBuffer.projOnly , float4(samplePos , 1.0f));
        // behind the camera : nothing to compare with
        if(clip.w <= 1e-4f) continue;

        const float2 ndc = clip.xy / clip.w;
        const float2 uv = float2(ndc.x * 0.5f + 0.5f , 0.5f - ndc.y * 0.5f);
        if(any(uv < 0.0f) || any(uv >= 1.0f)) continue;

        const uint2 coord = min(uint2(uv * float2(screenSize)) , screenSize - 1u);
        if(isSkyPixel(coord)) continue;

        const float3 occluderView = mul(generalBuffer.viewOnly , float4(position[coord].xyz , 1.0f)).xyz;

        // the surface that is seen at the pixel of the sample is in front of the sample ( closer to the camera = z closer to 0 )
        const float rangeCheck = smoothstep(0.0f , 1.0f , radius / max(abs(viewPos.z - occluderView.z) , 1e-4f));
        occlusion += (occluderView.z >= samplePos.z + 0.02f ? 1.0f : 0.0f) * rangeCheck;
    }

    return saturate(1.0f - occlusion / float(iterations));
}

// HBAO : for several directions on the screen the highest neighbour above the tangent plane of the pixel is looked for , the higher it
// is , the more the pixel is covered . the old version took atan2( dot , length( h - N ) * dot ) ( = about 45 degrees for any neighbour above the
// plane ) , sin() of a weight that already was a sin , a step of 0.01 of the screen for every depth and a radius that was not in pixels
float computeHBAO(uint2 currentPixel , float3 P , float3 N , float3 viewPos , float radius , uint iteration){
    const uint directions = max(min(iteration , 64u) , 1u);
    const uint steps = 6u;
    const int2 screenSize = int2(generalBuffer.screenWidth , generalBuffer.screenHeight);

    // the radius of the world in pixels at the distance of this pixel
    const float viewDepth = max(-viewPos.z , 1e-3f);
    const float pixelsPerUnit = 0.5f * float(generalBuffer.screenHeight) * generalBuffer.projOnly[1][1] / viewDepth;
    const float radiusPixels = clamp(radius * pixelsPerUnit , 4.0f , 160.0f);

    const float dither = pixelDither(currentPixel);

    float occlusionSum = 0.0f;
    for(uint d = 0 ; d < directions ; d++){
        const float angle = 2.0f * PI * (float(d) + dither) / float(directions);
        const float2 direction = float2(cos(angle) , sin(angle));

        float highest = 0.0f;
        for(uint s = 1u ; s <= steps ; s++){
            // the steps of neighbouring pixels are shifted by the dither
            const float t = (float(s) - dither) / float(steps);
            const int2 coord = int2(currentPixel) + int2(floor(direction * radiusPixels * t + 0.5f));
            if(any(coord < 0) || any(coord >= screenSize)) break;

            const uint2 ucoord = uint2(coord);
            if(isSkyPixel(ucoord)) continue;

            const float3 H = position[ucoord].xyz - P;
            const float dist = length(H);
            if(dist < 1e-4f) continue;

            // the sine of the angle above the tangent plane , a small bias keeps a flat floor clean
            const float elevation = saturate(dot(H / dist , N) - 0.1f);
            const float falloff = saturate(1.0f - (dist * dist) / (radius * radius));
            highest = max(highest , elevation * falloff);
        }
        occlusionSum += highest;
    }

    return saturate(1.0f - occlusionSum / float(directions));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    output.GetDimensions(width , height);
    uint2 screenSize = uint2(width , height);
    uint2 currentPixel = id.xy;
    if(any(currentPixel >= screenSize)) return;

    // the sky has no ambient occlusion
    if(isSkyPixel(currentPixel)){
        output[id.xy] = 1.0f;
        return;
    }

    float radius = max(features.AOradius , 0.01f);
    uint iteration = features.AO_iteration;
    float AO = 1.0f;
    uint mode = features.AO;

    const float3 P = position[id.xy].xyz;
    const float3 N = geometricNormal(currentPixel , P , normalize(normal[id.xy].xyz));

    const float3 viewPos = mul(generalBuffer.viewOnly , float4(P , 1.0f)).xyz;
    // N is a direction , not a point - w must be 0 so the translation of the view matrix is never applied to it
    const float3 N_view = normalize(mul(generalBuffer.viewOnly , float4(N , 0.0f)).xyz);

    if(mode == HBAO_REQUIRED){

        AO = computeHBAO(currentPixel , P , N , viewPos , radius , iteration);

    }else if( mode == SSAO_REQUIRED){

        AO = computeSSAO(currentPixel , viewPos , N_view , radius , iteration);

    }else{
        AO = 1.0f;
    }

    AO = pow(saturate(AO) , 2.0f);

    // a little of the last frame , at the place where this surface was seen then
    float4 prevPos = mul(generalBuffer.prevViewProj , float4(P , 1.0f) );
    float3 projCoord = prevPos.xyz / prevPos.w;
    float2 prevUV = projCoord.xy * 0.5f + 0.5f;
    prevUV.y = 1.0f - prevUV.y;

    const bool validUV = (prevPos.w > 0.0f && prevUV.x >= 0.0f && prevUV.x < 1.0f && prevUV.y >= 0.0f && prevUV.y < 1.0f);

    if(validUV){
        uint2 prevCoord = uint2((prevUV * float2(screenSize)) - 0.5f);
        const float prevAO = history[prevCoord];
        // 0 is an empty history ( a new image ) , not a fully occluded pixel
        if(prevAO > 0.0f){
            AO = lerp(prevAO , AO , 0.3f);
        }
    }

    output[id.xy] = AO;
}

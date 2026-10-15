#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::image_format("rgba16f")]]
[[vk::binding(0 , 0)]] RWTexture2D<float4> diffuseImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(1 , 0)]] RWTexture2D<float4> specularImage;

[[vk::image_format("r32f")]]
[[vk::binding(2 , 0)]] RWTexture2D<float> pixelCheck;

// the linear depth : the z distance from the camera in meters ( LinearizeDepth of the deferred render ) , it is NOT the distance along the ray
[[vk::image_format("r32f")]]
[[vk::binding(3 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(4 , 0)]] RWTexture2D<float4> indirectImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(5 , 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(6 , 0)]] RWTexture2D<float4> HDRImage;

[[vk::binding(7 , 0)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(8 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(9 , 0)]] ConstantBuffer<FogVolume> fogData;

[[vk::binding(10 , 0)]] SamplerState samp;
[[vk::binding(11 , 0)]] Texture2DArray fogImage;

// since the diffuse already contains shadows / AO  , the only thing we should do is multiply the diffuse with albedo and thats kinda all

// ---------------------------------------------------------------- the fog
// the froxel volume of fogCalc covers the z depth from FOG_NEAR to FOG_FAR ( ShaderStructure.h : the same numbers as in fogCalc ) , its slices are spread
// logarithmically . fogCalc writes the light that the slice scatters ( rgb ) and its extinction ( a ) , it does not accumulate : the accumulation along
// the ray is done here . behind FOG_FAR and in the sky the fog goes on with the formula of the height fog

// the sky has no depth : the fog is taken for a ray of this length ( z depth , meters )
static const float FOG_SKY_DISTANCE = 20000.0f;

// the z depth of the border of a slice ( k = 0 is the near border of the volume , k = sliceCount is the far one )
float fogSliceDepth(const float k , const float sliceCount){
    return FOG_NEAR * pow(FOG_FAR / FOG_NEAR , k / sliceCount);
}

// the optical depth ( the integral of the density ) of the height fog along a piece of a ray :
//      density( y ) = fogData.density * exp( -heightFallOff * max( y - fogBaseHeight , 0 ) )
// it is constant under the base height and falls exponentially above it , so the ray is cut where it crosses the base height and both pieces
// have a closed form : constant * length , and ( exp( -k y1 ) - exp( -k y2 ) ) / ( k * dy ) for the exponential one
// origin - where the piece starts , direction - normalized , rayLength - the length of the piece ( meters , along the ray )
float ComputeHeightFog(float3 origin , float3 direction , float rayLength){
    const float falloff = fogData.heightFallOff;
    const float density = fogData.density;
    const float y0 = origin.y - fogData.fogBaseHeight;
    const float dy = direction.y;

    // no falloff : the same density at every height
    if(falloff < 1e-6f) return density * rayLength;

    // a horizontal ray : the density does not change along it
    if(abs(dy) < 1e-5f){
        return density * exp(-falloff * max(y0 , 0.0f)) * rayLength;
    }

    // where the ray crosses the base height ( it can be behind the start or behind the end )
    const float tCross = clamp(-y0 / dy , 0.0f , rayLength);

    // the part under the base height and the part above it
    float underStart , underEnd , aboveStart , aboveEnd;
    if(dy > 0.0f){
        // goes up : under until the crossing , above after it
        underStart = 0.0f;   underEnd = tCross;
        aboveStart = tCross; aboveEnd = rayLength;
    }else{
        // goes down : above until the crossing , under after it
        aboveStart = 0.0f;   aboveEnd = tCross;
        underStart = tCross; underEnd = rayLength;
    }

    const float yStart = y0 + dy * aboveStart;
    const float yEnd = y0 + dy * aboveEnd;

    float opticalDepth = density * max(underEnd - underStart , 0.0f);
    opticalDepth += density * (exp(-falloff * max(yStart , 0.0f)) - exp(-falloff * max(yEnd , 0.0f))) / (falloff * dy);
    return max(opticalDepth , 0.0f);
}

// the fog between the camera and the thing that is linearDepth meters away ( z depth ) in the direction of rayDirection :
//      color of the thing * what passes through the fog + what the fog scatters on the way
// the froxel volume is accumulated slice by slice from the camera to the depth of the pixel : one step of a slice is
//      T *= exp( -sigmaT * length ) ,  S += T * ( scatter - scatter * exp( -sigmaT * length ) ) / sigmaT
float3 applyFog(float3 color , float linearDepth , float3 rayDirection , float2 uv){
    // the volume is made in the z depth , the ray of a pixel off the center of the screen is longer by 1 / cos of the angle to the axis of the camera
    const float cosAxis = max(dot(rayDirection , generalBuffer.playerDirection) , 1e-3f);
    const float rayScale = 1.0f / cosAxis;

    uint width , height , sliceCount , levels;
    fogImage.GetDimensions(0 , width , height , sliceCount , levels);

    float transmittance = 1.0f;
    float3 scattered = float3(0.0f , 0.0f , 0.0f);

    // 1 : the froxel volume , up to the pixel or up to the end of the volume
    const float volumeEnd = min(linearDepth , FOG_FAR);

    [loop]
    for(uint k = 0u ; k < sliceCount ; k++){
        const float near = fogSliceDepth(float(k) , float(sliceCount));
        if(near >= volumeEnd) break;
        const float far = min(fogSliceDepth(float(k + 1u) , float(sliceCount)) , volumeEnd);

        // the length of the ray inside of this slice
        const float length = (far - near) * rayScale;

        // the slice is read at its own layer : the layers of an array are not filtered between each other , the xy are filtered
        const float4 slice = fogImage.SampleLevel(samp , float3(uv , float(k)) , 0);
        const float sigmaT = slice.a;
        const float3 scatter = slice.rgb;

        const float stepTransmittance = exp(-sigmaT * length);
        const float3 stepScattered = (sigmaT > 1e-7f) ? (scatter - scatter * stepTransmittance) / sigmaT : scatter * length;

        scattered += transmittance * stepScattered;
        transmittance *= stepTransmittance;
    }

    float3 result = color * transmittance + scattered;

    // 2 : behind the volume ( and in the sky ) the fog goes on by the formula of the height fog , it has no lamps there : only its own glow
    if(linearDepth > FOG_FAR){
        const float3 origin = generalBuffer.playerPosition + rayDirection * (FOG_FAR * rayScale);
        const float rayLength = (linearDepth - FOG_FAR) * rayScale;

        const float opticalDepth = ComputeHeightFog(origin , rayDirection , rayLength);
        const float farTransmittance = exp(-opticalDepth);
        // the glow scatters albedo * color per unit of the extinction : the integral over the piece is albedo * color * ( 1 - T )
        const float3 farScattered = fogData.albedo * fogData.color * (1.0f - farTransmittance);

        result = result * farTransmittance + farScattered;
    }

    return result;
}

float3 ComputePixelRayDirection(uint2 pixelCoord, uint2 screenSize, float4x4 invViewProj , float3 playerPosition)
{
    float2 uv = (float2(pixelCoord) + 0.5) / float2(screenSize);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float4 farPoint = mul(invViewProj, float4(ndc, 1.0, 1.0));
    farPoint /= farPoint.w;

    return normalize(farPoint.xyz - playerPosition);
}


[shader("compute")]
[numthreads(8 , 8 ,1 )]
void main(uint3 id : SV_DispatchThreadID){
    uint width , height;
    HDRImage.GetDimensions(width , height);
    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    const float2 uv = (float2(id.xy) + 0.5f) / float2(screenSize);
    // the world direction of the ray of this pixel ( the y of the screen goes down , the one of the ndc goes up )
    const float3 worldDir = ComputePixelRayDirection(id.xy , screenSize , generalBuffer.invViewProj , generalBuffer.playerPosition);

    if(pixelCheck[id.xy] == SKYBOX_PIXEL || features.PT){
        float3 sky = diffuseImage[id.xy].xyz;
        // the fog covers the sky too , else the far things would be in the fog and the sky behind them would not : a seam on the horizon
        if(!features.PT && features.volumentricFog){
            sky = applyFog(sky , FOG_SKY_DISTANCE , worldDir , uv);
        }
        HDRImage[id.xy] = float4(sky , diffuseImage[id.xy].w);
        return ;
    }

    const float3 diffuse = diffuseImage[id.xy].xyz;
    const float3 specular = specularImage[id.xy].xyz;
    const float3 indirect = indirectImage[id.xy].xyz * generalBuffer.indirectWeight;
    const float3 albedo = max(albedoImage[id.xy].xyz , 0.02f);
    const float3 direct = (diffuse * albedo + specular);
    const float linearDepth = depthImage[id.xy].x;
    float3 result = direct + indirect;

    if(features.currentMode == INDIRECT_MODE){
        // a debug view : no fog on it
        HDRImage[id.xy] = indirectImage[id.xy];
        return;
    }

    if(features.volumentricFog){
        result = applyFog(result , linearDepth , worldDir , uv);
    }

    HDRImage[id.xy] = float4(result , 1.0f);
}

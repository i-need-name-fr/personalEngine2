#include "../../include/ShaderStructures/ShaderStructure.h"

[[vk::binding(0 , 0)]] SamplerState typicalSampler;
[[vk::binding(1 , 0)]] Texture1D multiScatterTexture;
[[vk::binding(2 , 0)]] TextureCube skybox;

[[vk::image_format("r32f")]]
[[vk::binding(3 , 0)]] RWTexture2D<float> pixelCheck;

// input images
[[vk::image_format("rgba32f")]]
[[vk::binding(4 , 0)]] RWTexture2D<float4> positionImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(5 , 0)]] RWTexture2D<float4> normalImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(6 , 0)]] RWTexture2D<float4> albedoImage;

[[vk::image_format("r32f")]]
[[vk::binding(7 , 0)]] RWTexture2D<float> depthImage;

[[vk::image_format("r32f")]]
[[vk::binding(8 , 0)]] RWTexture2D<float> roughnessImage;

[[vk::image_format("r32f")]]
[[vk::binding(9 , 0)]] RWTexture2D<float> metallicImage;

[[vk::image_format("r32f")]]
[[vk::binding(10 , 0)]] RWTexture2D<float> filteredShadowImage;
[[vk::image_format("r32f")]]
[[vk::binding(11 , 0)]] RWTexture2D<float> filteredAmbientOcclusion;
// required output

[[vk::image_format("rgba16f")]]
[[vk::binding(12 , 0)]] RWTexture2D<float4> diffuseImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(13 , 0)]] RWTexture2D<float4> specularImage;

[[vk::image_format("rg16f")]]
[[vk::binding(14 , 0)]] RWTexture2D<float2> motionVector;

[[vk::binding(17 , 0)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(18 , 0)]] ConstantBuffer<GeneralFeatures> features;

[[vk::binding(19 , 0)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;
[[vk::binding(20 , 0)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(21 , 0)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(22 , 0)]] RWStructuredBuffer<lg::LightSource> lightBuffer;

// the sun disc in the sky
[[vk::binding(23 , 0)]] ConstantBuffer<SunSource> sunBuffer;
[[vk::binding(24 , 0)]] ConstantBuffer<SkyViewData> skyBuffer;
[[vk::binding(25 , 0)]] Texture2D transmissionLUT;

// the ao map of the material : 1 where there is no map
[[vk::image_format("r32f")]]
[[vk::binding(26 , 0)]] RWTexture2D<float> objectAmbientImage;

float3 ComputePixelRayDirection(uint2 pixelCoord, uint2 screenSize, float4x4 invViewProj , float3 playerPosition)
{
    float2 uv = (float2(pixelCoord) + 0.5) / float2(screenSize);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float4 farPoint = mul(invViewProj, float4(ndc, 1.0, 1.0));
    farPoint /= farPoint.w;

    return normalize(farPoint.xyz - playerPosition);   
}

float calcMultiScatterFactor(float3 F0 , float roughness){
    // average of the three channels, not their sum
    const float F0_avr = (F0.x + F0.y + F0.z) / 3.0f;
    const float Ess = multiScatterTexture.SampleLevel(typicalSampler , roughness, 0).x;
    return 1.0f + F0_avr * (1.0f / max(Ess , 0.001f) - 1.0f);
}

void computeLight(GeneralObjectInfo info , float3 L , out float3 diffuse , out float3 specular){
    const float roughness = info.roughness;
    const float metallic = info.metallic;
    const float3 V = info.V;
    const float3 N = info.N;
    const float3 albedo = info.albedo;
    const float3 H = normalize(V + L);
    const float NdotV = max(dot(N , V) , 0.0001f);
    const float VdotH = max(dot(V , H) , 0.0001f);
    const float NdotH = max(dot(N , H) , 0.0001f);
    const float NdotL = max(dot(N , L) , 0.0001f);

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f), albedo , metallic);
    const float D = D_GGX(NdotH , roughness);
    const float3 F = F_fresnel(F0 , VdotH);
    const float G = G_smith(NdotV , NdotL , roughness);

    float3 directSpecular = (D * G * F) / max(4.0f * NdotL * NdotV , 0.0001f);
    float multiScatter = calcMultiScatterFactor(F0 , roughness);
    specular = directSpecular * multiScatter;

    const float3 kD = (1.0f - F) * (1.0f - metallic);
    diffuse = kD * albedo / PI;
}

void findLightIndices(uint2 currentPixel , out uint firstLight , out uint lightCount){
    firstLight = 0;
    lightCount = 0;

    uint2 tileCoord = currentPixel / forwardInfo.TILE_SIZE;
    uint tileIndex = tileCoord.x + tileCoord.y * forwardInfo.TILE_PER_ROW;

    // forward.hlsl's InterlockedAdd keeps counting past MAX_LIGHTS (it only stops WRITING there)
    lightCount = min(lightCountPerTile[tileIndex] , (uint)MAX_LIGHTS);
    firstLight = tileIndex * MAX_LIGHTS;
}

// the attenuation of a point light : the physical falloff times a window that reaches exactly 0 at lightRadius .
// without the window a light shone on EVERY pixel of the tiles it was listed in , far beyond its radius : the lit and the dark parts met
// on the edges of the Forward+ tiles ( hard vertical and horizontal borders ) and on the sphere of the radius
float lightAttenuation(lg::LightSource light , float lightDistance){
    const float falloff = 1.0f / (1.0f + light.Linear * lightDistance + light.Quadratic * lightDistance * lightDistance);
    const float ratio = saturate(lightDistance / max(light.radius , 1e-4f));
    float window = saturate(1.0f - ratio * ratio * ratio * ratio);
    window *= window;
    return falloff * window;
}

// ---------------------------------------------------------------- is the light covered ? ( the screen space ray , the same march as SSTraceRay of indirectCalc.hlsl )
// the shadow rays of the RT / PT / RayQuery / VSM make the shadow themselves : the screen space check works only when none of them is on , otherwise the light would be shadowed twice
// ( PostProcess multiplies the picture by the shadow image again )
static const uint SS_SHADOW_STEPS = 20u;
static const float SS_SHADOW_MAX_DISTANCE = 30.0f;
// how far behind the visible surface the ray may be and still count as a hit : the screen knows only the front of things
static const float SS_SHADOW_THICKNESS = 2.0f;

bool useScreenSpaceShadows(){
    return features.VSM == 0u && features.RayQuery == 0u && features.RT == 0u && features.PT == 0u;
}

// 1 - the light reaches the point , 0 - something on the screen covers it . dir goes FROM the point TO the light
float lightVisibility(const float3 P , const float3 N , const float3 dir , const float lightDistance){
    if(!useScreenSpaceShadows()) return 1.0f;
    // a surface that looks away from the light is not lit by it whatever is in the way : nothing to check
    if(dot(N , dir) <= 0.0f) return 1.0f;

    const uint2 screenSize = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    const float3 origin = offsetRay(P , N , generalBuffer.playerPosition);
    const float stepSize = min(lightDistance , SS_SHADOW_MAX_DISTANCE) / float(SS_SHADOW_STEPS);

    [loop]
    for(uint i = 1u ; i < SS_SHADOW_STEPS ; i++){   // step 0 is the pixel itself
        const float3 currentP = origin + dir * stepSize * float(i);
        const float4 clip = mul(generalBuffer.viewProj , float4(currentP , 1.0f));
        // behind the camera : no pixel there
        if(clip.w <= 1e-4f) break;
        const float2 uv = (clip.xy / clip.w) * float2(0.5f , -0.5f) + 0.5f;
        if(any(uv >= 1.0f) || any(uv < 0.0f)) break;
        const uint2 pixel = min(uint2(uv * float2(screenSize)) , screenSize - 1u);

        // the sky has no normal ( 0 ) : it covers nothing
        const float3 pN = normalImage[pixel].xyz;
        if(dot(pN , pN) < 0.5f) continue;
        // the surface that looks away from the ray : the ray went behind its front side ( the screen knows only the front ) , it is not a real cover
        if(dot(pN , dir) > 0.0f) continue;

        const float3 pP = positionImage[pixel].xyz;
        const float diff = length(currentP - generalBuffer.playerPosition) - length(pP - generalBuffer.playerPosition);
        if(diff > 0.02f && diff < SS_SHADOW_THICKNESS) return 0.0f;
    }
    return 1.0f;
}


// ---------------------------------------------------------------- the sun disc
static const float SUN_ANGULAR_RADIUS = 0.00465f;
static const float SUN_DISC_BRIGHTNESS = 100.0f;
static const float SUN_DISC_MAX = 60000.0f;

float3 sunTransmittanceToCamera(){
    const float u = skyBuffer.sunCosZenith * 0.5f + 0.5f;
    const float v = saturate(skyBuffer.cameraHeight / (ATMOSPHERE_RADIUS - PLANET_RADIUS));
    return transmissionLUT.SampleLevel(typicalSampler , float2(u , v) , 0).rgb;
}

float3 AddSunDisc(float3 skyColor , float3 rayDir){
    const float lightLength = length(sunBuffer.direction);
    // the light is the direction where the light goes : the sun is on the other side
    if(lightLength < 1e-5f) return skyColor;
    const float3 toSun = -sunBuffer.direction / lightLength;

    // nothing under the horizon
    if(rayDir.y < 0.0f) return skyColor;

    // the angle of one pixel from the projection ( projOnly[1][1] is 1 / tan( fov / 2 ) )
    const float pixelAngle = 2.0f / (max(generalBuffer.projOnly[1][1] , 1e-3f) * float(max(generalBuffer.screenHeight , 1u)));
    const float cosAngle = dot(rayDir , toSun);
    if(cosAngle < cos(SUN_ANGULAR_RADIUS + pixelAngle)) return skyColor;

    const float angle = acos(clamp(cosAngle , -1.0f , 1.0f));
    const float coverage = saturate((SUN_ANGULAR_RADIUS + 0.5f * pixelAngle - angle) / pixelAngle);

    const float3 disc = min(sunBuffer.diffuse * SUN_DISC_BRIGHTNESS * sunTransmittanceToCamera() , float3(SUN_DISC_MAX , SUN_DISC_MAX , SUN_DISC_MAX));
    return skyColor + disc * coverage;
}

float3 randomInLightRadius(inout uint seed  , lg::LightSource light , const float3 L){
    const float u = rand(seed) , v = rand(seed);
    const float z = 1.0f - 2 * u;
    const float phi = 2.0f * PI * v;
    const float r = sqrt(saturate(1.0f - z * z));
    const float x = cos(phi) * r;
    const float y = sin(phi) * r;
    const float R = light.sourceRadius;
    float3 offset = R * float3(x ,y ,z);
    // L goes FROM the point TO the lamp , the visible half of the sphere is the one that looks back at the point : the offsets that look the way of L ( dot > 0 )
    // are on the far side of the lamp , they are turned over ( it was dot < 0 : the far half was chosen )
    const float facing = dot(offset , L);
    if(facing > 0.0f) offset = -offset;
    return light.position + offset;
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){
    uint width , height;
    pixelCheck.GetDimensions(width , height);
    uint2 screenSize = uint2(width , height);
    uint2 currentPixel = id.xy;
    if(any(currentPixel >= screenSize)) return;
    if(pixelCheck[id.xy] == UNDEFINED_PIXEL){

        pixelCheck[id.xy] = SKYBOX_PIXEL;
        const float3 direction = ComputePixelRayDirection(currentPixel , screenSize , generalBuffer.invViewProj, generalBuffer.playerPosition);
        const float3 skyColor = skybox.SampleLevel(typicalSampler , direction , 0).xyz;
        diffuseImage[id.xy] = float4(AddSunDisc(skyColor , direction) , 1.0f);
    }else{

        const float3 P = positionImage[id.xy].xyz;
        const float3 N = normalImage[id.xy].xyz;
        const float depth = depthImage[id.xy].x;
        const float metallic = metallicImage[id.xy].x;
        const float roughness = roughnessImage[id.xy].x;
        const float3 albedo = albedoImage[id.xy].xyz;

        // the shadows and the occlusion are not applied here : PostProcess multiplies the picture by them ( after the denoisers )

        uint lightCount;
        uint lightOffset;

        findLightIndices(currentPixel  ,lightOffset , lightCount);

        const float2 uv = (float2(currentPixel) + 0.5f) / float2(screenSize);

        const float4 prevPos = mul(generalBuffer.prevViewProj , float4(P , 1.0f));
        float2 prevUV = ( prevPos.xy / prevPos.w) * 0.5f + 0.5f;
        prevUV.y = 1.0f - prevUV.y;

        GeneralObjectInfo info;
        info.albedo = albedo;
        info.metallic = metallic;
        info.roughness = roughness;
        info.N = N;
        info.P = P;
        info.V = normalize(generalBuffer.playerPosition - P);

        float3 diffuse = float3(0.0f , 0.0f , 0.0f);
        float3 specular = float3(0.0f , 0.0f , 0.0f);
        float3 ambient;

        // the lamps of the tile , the same sum as in generalHit.hlsl ( RT ) and PThit.hlsl ( PT ) : the point of the lamp is a random point of its sphere ( light.sourceRadius ) ,
        // the light is light.diffuse * attenuation * NdotL times the BRDF . the shadow of a lamp is not traced here ( the screen space check , or PostProcess with the shadow image )
        uint seed = initSeed(id.xy , generalBuffer.frameIndex);

        [loop]
        for(uint s = 0u ; s < lightCount ; s++){
            const lg::LightSource light = lightBuffer[tileLightIndices[lightOffset + s]];

            float3 L = light.position - P;
            L = normalize(L);
            const float3 lightPos = randomInLightRadius(seed , light , L);
            float3 dir = lightPos - P;
            const float d = length(dir);
            dir = normalize(dir);

            const float NdotL = dot(N , dir);
            const float reach = lightShadowReach(d , light.radius , features.lightShadowFade);
            if(reach <= 0.0f || NdotL <= 0.0f) continue;

            const float3 lightEmission = light.diffuse * lightAttenuation(light , d) * NdotL;
            float3 lightDiffuse = 0.0f;
            float3 lightSpecular = 0.0f;
            computeLight(info , dir , lightDiffuse , lightSpecular);

            const float visibility = lightVisibility(P , N , dir , d);
            diffuse += lightDiffuse * lightEmission * visibility;
            specular += lightSpecular * lightEmission * visibility;
        }

        // the sun
        {
            float3 sunDiffuse;
            float3 sunSpecular;
            const float3 L_sun = -normalize(sunBuffer.direction);
            const float NdotL = saturate(dot(N , L_sun));
            computeLight(info , L_sun , sunDiffuse , sunSpecular);
            sunDiffuse *=  NdotL * sunBuffer.diffuse;
            sunSpecular *= NdotL  * sunBuffer.diffuse;
            // the sun is at infinity : the ray goes as far as the screen space check allows
            const float sunVisibility = lightVisibility(P , N , L_sun , SS_SHADOW_MAX_DISTANCE);
            diffuse += sunDiffuse * sunVisibility;
            specular += sunSpecular * sunVisibility;
        }

        // the ao map of the material takes the DIRECT light only ( the lamps and the sun ) : the ambient light and the indirect light are not here , and the SSAO / HBAO of the
        // post process is a different thing ( it knows nothing about the paint of the object )
        {
            // the strength is the same slider as the one of the SSAO / HBAO ( features.AO_intens ) : the same pow as in PostProcess
            const float objectAmbient = pow(objectAmbientImage[id.xy].x , features.AO_intens);
            diffuse *= objectAmbient;
            specular *= objectAmbient;
        }

        // no weight of the direct light here for now ( directWeight is not applied : the clean colour )

        // demodulation : PostProcess multiplies the denoised diffuse by the same max( albedo , 0.02 )
        diffuse /= max(albedo , 0.02f);

        // a single pixel can still be far brighter than its neighbours : cap it , it would smear into the
        // whole neighbourhood in the denoisers and in the bloom
        {
            const float maxRadiance = 24.0f;
            const float peak = max(max(diffuse.x + specular.x , diffuse.y + specular.y) , diffuse.z + specular.z);
            if(peak > maxRadiance){
                const float scale = maxRadiance / peak;
                diffuse *= scale;
                specular *= scale;
            }
        }

        const float3 V = info.V;
        const float NdotV = max(dot(N , V) , 0.0001f);

        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f), albedo , metallic);
        const float3 F = F_fresnel(F0 , NdotV);

        diffuseImage[id.xy] = float4(diffuse , 1.0f);
        specularImage[id.xy] = float4(specular , 1.0f);
        motionVector[id.xy] = prevUV - uv ;
    }
}
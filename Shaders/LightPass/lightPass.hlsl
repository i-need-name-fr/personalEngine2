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

// we wont change it since , it requires to write in it
// ping-pong : this frame only WRITES ReSTIRDI_buffer (own pixel) and only READS prevReSTIRDI_buffer
// (temporal + spatial reuse) - reading neighbours from the buffer being written raced with the
// other invocations of this same dispatch
[[vk::binding(15 , 0)]] RWStructuredBuffer<ReservoirDI> ReSTIRDI_buffer;
[[vk::binding(16 , 0)]] RWStructuredBuffer<ReservoirDI> prevReSTIRDI_buffer;

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

uint bufferIndex(uint2 pixel){
    const uint2 clamped = min(pixel , uint2(generalBuffer.screenWidth - 1u , generalBuffer.screenHeight - 1u));
    return clamped.x + clamped.y * generalBuffer.screenWidth;
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

// ReSTIR DI is switched by features.ReSTIR ( ImGui ) : there is nothing to recompile

float targetPdf_DI(uint lightIndex , GeneralObjectInfo info){
    const lg::LightSource candidate = lightBuffer[lightIndex];
    float3 L = candidate.position - info.P;
    const float d = length(L);
    L /= max(d , 1e-4f);

    const float attenuation = lightAttenuation(candidate , d);
    float3 diffuse;
    float3 specular;

    computeLight(info , L , diffuse ,specular);
    const float3 brdfValue = (diffuse + specular) * max(dot(info.N , L) , 0.0001f);
    return luminance(candidate.diffuse * brdfValue * attenuation);
}

void finalizeReservoir_DI(inout ReservoirDI r , GeneralObjectInfo info){
    // an undefined light has no p_hat : lightBuffer must not be read with this index
    if(r.chosenIndex == 9999u){
        r.W = 0.0f;
        return;
    }
    const float p_hat = targetPdf_DI(r.chosenIndex , info);
    r.W = (p_hat > 1e-6f && r.M > 0u) ? r.weightSum / (float(r.M) * p_hat) : 0.0f;
}

void generateReservoir_DI(inout ReservoirDI res , inout uint seed , GeneralObjectInfo info, uint lightIndicesCount , uint lightOffset){
    res.W = 0.0f;
    res.M = 0;
    res.weightSum = 0.0f;
    res.chosenIndex = 9999u;   // undefined : no light is chosen
    res.diffuseColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
    res.specularColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
    if(lightIndicesCount == 0u) return;

    float bestPdf = 0.0f;
    for(uint i = 0 ; i < lightIndicesCount ; i++){
        const uint lightIndex = tileLightIndices[lightOffset + i];
        const lg::LightSource light = lightBuffer[lightIndex];

        float3 toLight = light.position - info.P;
        const float lightDistance = length(toLight);
        toLight /= max(lightDistance , 1e-4f);

        float3 candidateDiffuse;
        float3 candidateSpecular;
        computeLight(info , toLight , candidateDiffuse , candidateSpecular);

        const float3 radiance = light.diffuse * lightAttenuation(light , lightDistance) * max(dot(info.N , toLight) , 0.0001f);
        candidateDiffuse *= radiance;
        candidateSpecular *= radiance;

        res.diffuseColor.xyz += candidateDiffuse;
        res.specularColor.xyz += candidateSpecular;

        const float p_hat = luminance(candidateDiffuse + candidateSpecular);
        res.weightSum += p_hat;
        // a light wins only with p_hat > 0
        if(p_hat > bestPdf){
            bestPdf = p_hat;
            res.chosenIndex = lightIndex;
        }
    }

    // nothing reaches the pixel : no light is chosen ( M = 0 , W = 0 ) , and finalizeReservoir_DI is not called with an undefined index
    if(res.chosenIndex == 9999u) return;

    res.M = 1u;
    finalizeReservoir_DI(res , info);
}

ReservoirDI combineReservoir_DI(ReservoirDI a , ReservoirDI b , inout uint seed){
    ReservoirDI result;
    if(a.chosenIndex == 9999) return b;
    if(b.chosenIndex == 9999) return a;

    result.M = a.M + b.M;
    result.weightSum = a.weightSum + b.weightSum;
    result.chosenIndex = a.chosenIndex;
    result.W = 0.0f;   // recomputed by finalizeReservoir_DI

    if(result.weightSum > 0.0f && rand(seed) * result.weightSum < b.weightSum){
        result.chosenIndex = b.chosenIndex;
    }
    return result;
}

void temporalReuse_DI(inout ReservoirDI w , inout uint seed , uint2 prevPixel, GeneralObjectInfo info){
    const ReservoirDI prev = prevReSTIRDI_buffer[bufferIndex(prevPixel)];
    if(prev.M == 0u) return;

    const uint M_clamped = min(prev.M , 20u * max(w.M , 1u));

    ReservoirDI pRes;
    pRes.M = M_clamped;
    pRes.chosenIndex = prev.chosenIndex;
    if(prev.chosenIndex == 9999) return;
    // canonical ReSTIR weight of a reused sample: p_hat_here * W * M
    pRes.weightSum = prev.W * float(M_clamped) * targetPdf_DI(prev.chosenIndex , info);
    pRes.W = 0.0f;

    w = combineReservoir_DI(w , pRes , seed);
    finalizeReservoir_DI(w ,info);
}

void spatialReuse_DI(inout ReservoirDI r , inout uint seed , uint2 currentPixel, GeneralObjectInfo info){
    const uint numSpatialSamples = 5u;
    const float spatialRadius = 30.0f;   // in pixels

    for(uint s = 0u; s < numSpatialSamples; s++){
        const float rand1 = rand(seed);
        const float rand2 = rand(seed);
        const float phi = 2.0f * PI * rand2;

        const float radius = sqrt(rand1) * spatialRadius;
        const int2 neighborPixel = int2(currentPixel) + int2(cos(phi) * radius , sin(phi) * radius);
        // signed compare - casting to uint first made the "< 0" test dead
        if(neighborPixel.x < 0 || neighborPixel.y < 0 ||
           neighborPixel.x >= (int)generalBuffer.screenWidth || neighborPixel.y >= (int)generalBuffer.screenHeight){
            continue;
        }

        const ReservoirDI stored = prevReSTIRDI_buffer[bufferIndex((uint2)neighborPixel)];
        if(stored.M == 0u || stored.W <= 0.0f) continue;
        if(stored.chosenIndex == 9999) continue;
        // Geometric rejection used to read normalOutputImage/depthOutputImage of
        // the neighbour, which other invocations of this very dispatch are still
        // writing - a data race. Re-evaluating the target function at THIS pixel
        // rejects unusable neighbours without touching the shared G-buffer.
        const float p_hat = targetPdf_DI(stored.chosenIndex , info);
        if(p_hat <= 0.0f) continue;

        ReservoirDI neighborRes;
        neighborRes.chosenIndex = stored.chosenIndex;
        neighborRes.M = stored.M;
        neighborRes.W = 0.0f;
        // the stored.W factor was missing here while temporalReuse_DI had it
        neighborRes.weightSum = stored.W * float(stored.M) * p_hat;

        r = combineReservoir_DI(r , neighborRes , seed);
        finalizeReservoir_DI(r ,info);
    }
}

ReservoirDI ReSTIR_DI_generation(uint2 currentPixel , float2 prevUV , inout uint seed , GeneralObjectInfo info,  uint lightIndicesCount , uint lightOffset){
    const bool validUV = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    const uint2 prevCoords = (uint2)clamp(prevUV * float2(generalBuffer.screenWidth , generalBuffer.screenHeight) ,
                                          float2(0.0f , 0.0f) ,
                                          float2(generalBuffer.screenWidth - 1u , generalBuffer.screenHeight - 1u));

    // every light is visited in generateReservoir_DI , so there is nothing to collect from the last frame or from the neighbours . ( the reuse picks between
    // the reservoirs by a random number : it would bring the flashes back )
    ReservoirDI r;
    generateReservoir_DI(r , seed , info , lightIndicesCount , lightOffset);
    const uint M_CAP = 500u;
    if(r.M > M_CAP){
        r.weightSum *= float(M_CAP) / float(r.M);
        r.M = M_CAP;
    }
    return r;
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

        // otherwise next frame's reuse would read whatever this slot held two frames ago
        ReservoirDI empty;
        empty.M = 0u;
        empty.chosenIndex = 9999u;
        empty.W = 0.0f;
        empty.weightSum = 0.0f;
        empty.diffuseColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
        empty.specularColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
        ReSTIRDI_buffer[bufferIndex(currentPixel)] = empty;
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

        ReservoirDI reservoir;
        reservoir.M = 0u;
        reservoir.chosenIndex = 9999u;   // undefined : no light is chosen
        reservoir.W = 0.0f;
        reservoir.weightSum = 0.0f;
        reservoir.diffuseColor = float4(0.0f , 0.0f , 0.0f , 0.0f);
        reservoir.specularColor = float4(0.0f , 0.0f , 0.0f , 0.0f);

        float3 L = normalize(info.V + N);

        if(features.ReSTIR != 0u){

            // frameIndex * x + y gave the same seed to whole diagonals of pixels ( correlated noise ) : a hash of the pixel and the frame
            uint seed = initSeed(id.xy , generalBuffer.frameIndex);

            reservoir = ReSTIR_DI_generation(currentPixel, prevUV ,  seed , info , lightCount , lightOffset );

            // the light of the lamps is the sum of all the candidates : the reservoir keeps it
            diffuse = reservoir.diffuseColor.xyz;
            specular = reservoir.specularColor.xyz;

            if(reservoir.chosenIndex != 9999){
                L = normalize(lightBuffer[reservoir.chosenIndex].position - P);
            }

        }else{

            // no random at all : every light of the tile is added up . the result of a pixel does not depend on a chosen light or on the
            // frame , so nothing can jump from one frame to the next ( the random choice of a light flipped at the edges of the
            // lights and of the shadows ) . the number of the lights of a tile is small , and it is capped by MAX_LIGHTS
            float dominantLuminance = 0.0f;
            uint dominantLight = 0u;

            for(uint s = 0u ; s < lightCount ; s++){
                const uint lightIndex = tileLightIndices[lightOffset + s];
                const lg::LightSource light = lightBuffer[lightIndex];

                const float3 toLight = light.position - P;
                const float lightDistance = length(toLight);
                const float3 lightDir = toLight / max(lightDistance , 1e-4f);

                float3 lightDiffuse;
                float3 lightSpecular;
                computeLight(info , lightDir , lightDiffuse , lightSpecular);

                const float attenuation = lightAttenuation(light , lightDistance);
                float3 radiance = light.diffuse * attenuation * max(dot(N , lightDir) , 0.0001f);


                diffuse += lightDiffuse * radiance;
                specular += lightSpecular * radiance;

                // before the shadow : the light that matters the most for this pixel
                const float contributionLuminance = luminance((lightDiffuse + lightSpecular) * radiance);
                if(contributionLuminance > dominantLuminance){
                    dominantLuminance = contributionLuminance;
                    dominantLight = lightIndex;
                }
            }

            // without ReSTIR nobody chooses a light , but the shadow pass ( RayQueryShadows ) reads the chosen light from the reservoir of the previous frame :
            // it was always empty ( the light 0 ) , so the shadow was traced to the light 0 only . now it is the dominant light of the pixel
            if(dominantLuminance > 0.0f){
                reservoir.chosenIndex = dominantLight;
                reservoir.M = 1u;
                reservoir.W = 1.0f;
                reservoir.weightSum = dominantLuminance;
            }
        }

        // the sun : it does not depend on the choice of a lamp , so it is added after both branches ( it was inside of the ReSTIR branch only )
        {
            float3 sunDiffuse;
            float3 sunSpecular;
            const float3 L_sun = -normalize(sunBuffer.direction);
            const float NdotL = saturate(dot(N , L_sun));
            computeLight(info , L_sun , sunDiffuse , sunSpecular);
            sunDiffuse *=  NdotL * sunBuffer.diffuse;
            sunSpecular *= NdotL  * sunBuffer.diffuse;
            diffuse += sunDiffuse;
            specular += sunSpecular;
        }

        // the ao map of the material takes the DIRECT light only ( the lamps and the sun ) : the ambient light and the indirect light are not here , and the SSAO / HBAO of the
        // post process is a different thing ( it knows nothing about the paint of the object ) . the reservoir keeps the light without it
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

        ReSTIRDI_buffer[bufferIndex(currentPixel)] = reservoir;
        diffuseImage[id.xy] = float4(diffuse , 1.0f);
        specularImage[id.xy] = float4(specular , 1.0f);
        motionVector[id.xy] = prevUV - uv ;
    }
}
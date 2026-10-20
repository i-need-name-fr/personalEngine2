float RadicalInverse_VdC(uint bits) 
{ 
	bits = (bits << 16u) | (bits >> 16u); 
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u); 
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u); 
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u); 
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u); 
	return float(bits) * 2.3283064365386963e-10; // делим на 2^32 
}


float2 Hammersley(uint i, uint N) { 
	return float2(float(i) / float(N), RadicalInverse_VdC(i)); 
}


float G1_schlick(float NdotX , float k){
    return NdotX / (NdotX * (1.0f - k) + k);
}

float G_smith(float NdotV , float NdotL , float roughness){
    const float k_ambient = (roughness) * (roughness) / 2.0f;
    const float G1_L = G1_schlick(NdotL , k_ambient );
    const float G1_V = G1_schlick(NdotV , k_ambient );
    return G1_L * G1_V;
}

static const float PI = 3.14159265f;

float3 GGX_halfVector(float2 Xi , float3 normal , float roughness){
    float rand1 = Xi.x;
    float rand2 = Xi.y;

    const float phi = 2.0f * PI * rand1;
    float a = roughness * roughness;
    float a2 = a * a ;
    const float cosTheta = sqrt(saturate((1.0f - rand2) / max(1.0f + (a2 - 1.0f) * rand2 , 1e-6f)));
    const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    const float3 up = (abs(normal.y) < 0.999f ? float3(0.0f , 1.0f , 0.0f) : float3(1.0f , 0.0f , 0.0f));
    // cross() is only unit length when up and normal are orthogonal - normalise
    // so the basis stays orthonormal for any normal
    const float3 tangent = normalize(cross(up , normal));
    const float3 bitangent = cross(normal , tangent);
    return normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * normal);
}

float computeEss(float roughness){
    const uint NdotV_samples = 32u;
    const uint SAMPLE_COUNT = 1024u;

    float totalEnerty = 0.0f;

    for(uint v = 0 ; v < NdotV_samples ; v++){

        const float NdotV = (float(v) + 0.5f) / (float(NdotV_samples));
        float3 V = float3(sqrt(1.0f - NdotV * NdotV) , 0.0f , NdotV);
        float3 N = float3(0.0f , 0.0f , 1.0f);

        float energyForThisV = 0.0f;

        for(uint i = 0 ; i < SAMPLE_COUNT ; i++){

            float2 Xi = Hammersley(i , SAMPLE_COUNT);
            float3 H = GGX_halfVector(Xi , N , roughness);
            float3 L = normalize(2.0f * dot(V , H) * H - V);

            const float NdotL = max(L.z , 0.0f);
            if(NdotL > 0.0f){
                float G = G_smith(NdotV  , NdotL , roughness);
                float VdotH = max(dot(V , H) , 0.0f);
                float NdotH = max(H.z , 0.0f);
                energyForThisV += (G * VdotH) / max(NdotH * NdotV , 0.0001f);
            }
        }
        totalEnerty += energyForThisV / float(SAMPLE_COUNT);
    }

    return totalEnerty / float(NdotV_samples);
}
[[vk::image_format("r32f")]]
[[vk::binding( 0 , 0)]] RWTexture1D<float> result;

[shader("compute")]
[numthreads(64, 1 , 1)]
void main(uint3 id : SV_DispatchThreadID){
    uint lutSize;
    result.GetDimensions(lutSize);
    if (id.x >= lutSize) return;

    float roughness = (float(id.x) + 0.5) / float(lutSize);
    result[id.x] = computeEss(roughness);
}
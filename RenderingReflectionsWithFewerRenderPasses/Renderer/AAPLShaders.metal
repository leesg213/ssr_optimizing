/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Metal shaders used for this sample.
*/

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

#import "AAPLShaderTypes.h"

typedef struct
{
    float3 position  [[attribute(VertexAttributePosition)]];
    float2 texCoord  [[attribute(VertexAttributeTexcoord)]];
    half3  normal    [[attribute(VertexAttributeNormal)]];
    half3  tangent   [[attribute(VertexAttributeTangent)]];
    half3  bitangent [[attribute(VertexAttributeBitangent)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;

    half3  worldPos;
    half3  tangent;
    half3  bitangent;
    half3  normal;
} ColorInOut;

// Vertex function
vertex ColorInOut vertexTransform (const Vertex in                               [[ stage_in ]],
                                   const uint   instanceId                       [[ instance_id ]],
                                   const device InstanceParams* instanceParams   [[ buffer     (BufferIndexInstanceParams) ]],
                                   const device ActorParams&    actorParams      [[ buffer (BufferIndexActorParams)    ]],
                                   constant     ViewportParams* viewportParams   [[ buffer (BufferIndexViewportParams) ]] )
{
    ColorInOut out;
    out.texCoord = in.texCoord;

    float4x4 modelMatrix = actorParams.modelMatrix;
    float4x4 viewMatrix = viewportParams[0].viewMatrix;
    float4 worldPos  = modelMatrix * float4(in.position, 1.0);
    float4 screenPos = viewportParams[0].viewProjectionMatrix * worldPos;

    out.worldPos = (half3)in.position.xyz;
    out.position = screenPos;

	float4x4 normalMatrix = modelMatrix;//float4x4(viewMatrix * modelMatrix);
    
    float4 vTangent = normalMatrix * float4(float3(in.tangent.xyz), 0);
    float4 vBitangent = normalMatrix * float4(float3(in.bitangent.xyz), 0);
    float4 vNormal = normalMatrix * float4(float3(in.normal.xyz), 0);

    out.tangent   = half3(vTangent.xyz);
    out.bitangent = half3(vBitangent.xyz);
    out.normal    = half3(vNormal.xyz);

    return out;
}

struct GBufferOut
{
    float4 normal_refl_mask [[color(0)]];
    float4 diffuse [[color(1)]];
};

fragment GBufferOut fragmentGBuffer(         ColorInOut      in             [[ stage_in ]],
                         device   ActorParams&    actorParams    [[ buffer (BufferIndexActorParams)    ]],
                         constant FrameParams &   frameParams    [[ buffer (BufferIndexFrameParams)    ]],
                         constant ViewportParams* viewportParams [[ buffer (BufferIndexViewportParams) ]],
                                  texture2d<half> baseColorMap   [[ texture (TextureIndexBaseColor)    ]],
                                  texture2d<half> normalMap      [[ texture (TextureIndexNormal)       ]],
                                  texture2d<half> specularMap    [[ texture (TextureIndexSpecular)     ]] )
{
    GBufferOut out;
    
    float4 normal_refl_mask = 0;
    float4 diffuse = 0;
    
    constexpr sampler linearSampler (mip_filter::linear,
                                     mag_filter::linear,
                                     min_filter::linear);

    const half4 baseColorSample = baseColorMap.sample (linearSampler, in.texCoord.xy);
    half3 normalSampleRaw = normalMap.sample (linearSampler, in.texCoord.xy).xyz;
    // The x and y coordinates in a normal map (red and green channels) are mapped from [-1;1] to [0;255].
    // As the sampler returns a value in [0 ; 1], we need to do :
    normalSampleRaw.xy = normalSampleRaw.xy * 2.0 - 1.0;
    const half3 normalSample = normalize(normalSampleRaw);

    // The per-vertex vectors have been interpolated, thus we need to normalize them again :
    in.tangent   = normalize (in.tangent);
    in.bitangent = normalize (in.bitangent);
    in.normal    = normalize (in.normal);

    half3x3 tangentMatrix = half3x3(in.tangent, in.bitangent, in.normal);

    normal_refl_mask.xyz = (float3) (tangentMatrix * normalSample);
    normal_refl_mask.w = 0; // No Refelction
    
    diffuse.xyz = float3(baseColorSample.xyz) * actorParams.diffuseMultiplier;
    
    out.normal_refl_mask = normal_refl_mask;
    out.diffuse = diffuse;
    
    return out;
}

// Fragment function used to render the temple object in both the
//   reflection pass and the final pass
fragment float4 fragmentLighting (         ColorInOut      in             [[ stage_in ]],
                                  device   ActorParams&    actorParams    [[ buffer (BufferIndexActorParams)    ]],
                                  constant FrameParams &   frameParams    [[ buffer (BufferIndexFrameParams)    ]],
                                  constant ViewportParams* viewportParams [[ buffer (BufferIndexViewportParams) ]],
                                           texture2d<half> baseColorMap   [[ texture (TextureIndexBaseColor)    ]],
                                           texture2d<half> normalMap      [[ texture (TextureIndexNormal)       ]],
                                           texture2d<half> specularMap    [[ texture (TextureIndexSpecular)     ]] )
{
    constexpr sampler linearSampler (mip_filter::linear,
                                     mag_filter::linear,
                                     min_filter::linear);

    const half4 baseColorSample = baseColorMap.sample (linearSampler, in.texCoord.xy);
    half3 normalSampleRaw = normalMap.sample (linearSampler, in.texCoord.xy).xyz;
    // The x and y coordinates in a normal map (red and green channels) are mapped from [-1;1] to [0;255].
    // As the sampler returns a value in [0 ; 1], we need to do :
    normalSampleRaw.xy = normalSampleRaw.xy * 2.0 - 1.0;
    const half3 normalSample = normalize(normalSampleRaw);

    const half  specularSample  = specularMap.sample  (linearSampler, in.texCoord.xy).x*0.5;

    // The per-vertex vectors have been interpolated, thus we need to normalize them again :
    in.tangent   = normalize (in.tangent);
    in.bitangent = normalize (in.bitangent);
    in.normal    = normalize (in.normal);

    half3x3 tangentMatrix = half3x3(in.tangent, in.bitangent, in.normal);

    float3 normal = (float3) (tangentMatrix * normalSample);

    float3 directionalContribution = float3(0);
    float3 specularTerm = float3(0);
    {
        float nDotL = saturate (dot(normal, frameParams.directionalLightInvDirection));

        // The diffuse term is the product of the light color, the surface material
        // reflectance, and the falloff
        float3 diffuseTerm = frameParams.directionalLightColor * nDotL;

        // Apply specular lighting...

        // 1) Calculate the halfway vector between the light direction and the direction they eye is looking
        float3 eyeDir = normalize (viewportParams[0].cameraPos - float3(in.worldPos));
        float3 halfwayVector = normalize(frameParams.directionalLightInvDirection + eyeDir);

        // 2) Calculate the reflection amount by evaluating how the halfway vector matches the surface normal
        float reflectionAmount = saturate(dot(normal, halfwayVector));

        // 3) Calculate the specular intensity by powering our reflection amount to our object's
        //    shininess
        float specularIntensity = powr(reflectionAmount, actorParams.materialShininess);

        // 4) Obtain the specular term by multiplying the intensity by our light's color
        specularTerm = frameParams.directionalLightColor * specularIntensity * float(specularSample);

        // The base color sample is actually the diffuse color of the material
        float3 baseColor = float3(baseColorSample.xyz) * actorParams.diffuseMultiplier;

        // The ambient contribution is an approximation for global, indirect lighting, and simply added
        //   to the calculated lit color value below

        // Calculate diffuse contribution from this light : the sum of the diffuse and ambient * albedo
        directionalContribution = baseColor * (diffuseTerm + frameParams.ambientLightColor);
    }

    // Now that we have the contributions our light sources in the scene, we sum them together
    //   to get the fragment's lit color value
    float3 color = specularTerm + directionalContribution;

    // We return the color we just computed and the alpha channel of our baseColorMap for this
    //   fragment's alpha value
    return float4(color, baseColorSample.w);
}


fragment float4 fragmentGround (         ColorInOut      in             [[ stage_in ]],
                                constant ViewportParams* viewportParams [[ buffer (BufferIndexViewportParams) ]] )
{
    float onEdge;
    {
        float2 onEdge2d = fract(float2(in.worldPos.xz)/500.f);
        // If onEdge2d is negative, we want 1. Otherwise, we want zero (independent for each axis).
        float2 offset2d = sign(onEdge2d) * -0.5 + 0.5;
        onEdge2d += offset2d;
        onEdge2d = step (0.03, onEdge2d);

        onEdge = min(onEdge2d.x, onEdge2d.y);
    }

    float3 neutralColor = float3 (0.9, 0.9, 0.9);
    float3 edgeColor = neutralColor * 0.2;
    float3 groundColor = mix (edgeColor, neutralColor, onEdge);

    return float4 (groundColor, 1.0);
}


fragment GBufferOut fragmentGroundGBuffer(         ColorInOut      in             [[ stage_in ]] )
{
    GBufferOut out;
    
    float onEdge;
    {
        float2 onEdge2d = fract(float2(in.worldPos.xz)/500.f);
        // If onEdge2d is negative, we want 1. Otherwise, we want zero (independent for each axis).
        float2 offset2d = sign(onEdge2d) * -0.5 + 0.5;
        onEdge2d += offset2d;
        onEdge2d = step (0.03, onEdge2d);

        onEdge = min(onEdge2d.x, onEdge2d.y);
    }

    float3 neutralColor = float3 (0.9, 0.9, 0.9) * 0.25;
    float3 edgeColor = neutralColor * 0.2;
    float3 groundColor = mix (edgeColor, neutralColor, onEdge);
    
    float4 normal_refl_mask = 0;
    float4 diffuse = 0;
    
    normal_refl_mask.xyz = normalize(float3(in.normal));
    normal_refl_mask.w = 1;
    diffuse.xyz = float3(0,0,1);
    
    out.normal_refl_mask = normal_refl_mask;
    out.diffuse = float4(groundColor, 1);
    
    return out;
}


fragment GBufferOut fragmentWallGBuffer(         ColorInOut      in             [[ stage_in ]] )
{
    GBufferOut out;
    
    float onEdge;
    {
        float2 onEdge2d = min(fract(float2(in.worldPos.xy)/(250.f/8)),fract(float2(in.worldPos.xz)/(250.f/8)));
        // If onEdge2d is negative, we want 1. Otherwise, we want zero (independent for each axis).
        float2 offset2d = sign(onEdge2d) * -0.5 + 0.5;
        onEdge2d += offset2d;
        onEdge2d = step (0.03, onEdge2d);

        onEdge = min(onEdge2d.x, onEdge2d.y);
    }

    float3 neutralColor = float3(in.normal)*0.5 + 0.5;
    float3 edgeColor = neutralColor * 0.2;
    float3 groundColor = mix (edgeColor, neutralColor, onEdge);
    
    float4 normal_refl_mask = 0;
    float4 diffuse = 0;
    
    normal_refl_mask.xyz = normalize(float3(in.normal));
    normal_refl_mask.w = 1;
    
    out.normal_refl_mask = normal_refl_mask;
    out.diffuse = float4(groundColor, 1);
    
    return out;
}

kernel void kernel_lighting(texture2d<float, access::read> normal_buffer [[texture(0)]],
                     texture2d<float, access::read> diffuse_buffer [[texture(1)]],
                     texture2d<float, access::write> output [[texture(2)]],
                     uint2 tid [[thread_position_in_grid]])
{
    
    float3 vLight = normalize(float3(1,1,0));
    float ambientIntensity = 0.3f;
    float3 normal = normal_buffer.read(tid).xyz;
    float directionalLightingIntensity = saturate(dot(normal, vLight));
    float4 diffuse = diffuse_buffer.read(tid);
    
    float4 finalColor = 0;
    
    if(diffuse.x == 0 && diffuse.y == 0 && diffuse.z == 0)
        finalColor = float4(0,0,1, 1);
    else
        finalColor = diffuse * directionalLightingIntensity + diffuse * ambientIntensity;
    
    output.write(pow(finalColor,2.2f), tid);
}


// Screen filling quad in normalized device coordinates
constant float2 quadVertices[] = {
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

struct CopyVertexOut {
    float4 position [[position]];
    float2 uv;
};

// Simple vertex shader which passes through NDC quad positions
vertex CopyVertexOut copyVertex(unsigned short vid [[vertex_id]]) {
    float2 position = quadVertices[vid];
    
    CopyVertexOut out;
    
    out.position = float4(position, 0, 1);
    out.uv = position * 0.5f + 0.5f;
    out.uv.y = 1 - out.uv.y;
    return out;
}

// Simple fragment shader which copies a texture and applies a simple tonemapping function
fragment float4 copyFragment(CopyVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float3 color = tex.sample(sam, in.uv).xyz;
    
    // Apply a very simple tonemapping function to reduce the dynamic range of the
    // input image into a range which can be displayed on screen.
    color = color / (1.0f + color);
    
    return float4(color, 1.0f);
}

kernel void kernel_depthCopy(texture2d<float, access::read> depth [[ texture(0) ]],
                           texture2d<float, access::write> outDepth [[ texture(1) ]],
                                  uint2 tid [[thread_position_in_grid]])
{
    outDepth.write(depth.read(tid).x, tid);
}

kernel void kernel_createHiZ(texture2d<float, access::read> depth [[ texture(0) ]],
                           texture2d<float, access::write> outDepth [[ texture(1) ]],
                                  uint2 tid [[thread_position_in_grid]])
{
	float2 depth_dim = float2(depth.get_width(), depth.get_height());
	float2 out_depth_dim = float2(outDepth.get_width(), outDepth.get_height());
	
	float2 ratio = depth_dim / out_depth_dim;
	
	uint2 vReadCoord = tid<<1;//uint2(floor(float2(tid) * ratio));
    uint2 vWriteCoord = tid;
    
    float4 depth_samples = float4(
                                  depth.read(vReadCoord).x,
                                  depth.read(vReadCoord + uint2(1,0)).x,
                                  depth.read(vReadCoord + uint2(0,1)).x,
                                  depth.read(vReadCoord + uint2(1,1)).x
                                  );
	
    float min_depth = min(depth_samples.x, min(depth_samples.y, min(depth_samples.z, depth_samples.w)));
    
	bool needExtraSampleX = ratio.x>2;
    bool needExtraSampleY = ratio.y>2;
    
    min_depth = needExtraSampleX ? min(min_depth, min(depth.read(vReadCoord + uint2(2,0)).x, depth.read(vReadCoord + uint2(2,1)).x)) : min_depth;
    min_depth = needExtraSampleY ? min(min_depth, min(depth.read(vReadCoord + uint2(0,2)).x, depth.read(vReadCoord + uint2(1,2)).x)) : min_depth;
    min_depth = (needExtraSampleX && needExtraSampleY) ? min(min_depth, depth.read(vReadCoord + uint2(2,2)).x) : min_depth;
    
    outDepth.write(float4(min_depth), vWriteCoord);
    
}

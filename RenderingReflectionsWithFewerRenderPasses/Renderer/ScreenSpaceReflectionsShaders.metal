//
//  ScreenSpaceReflectionsShaders.metal
//  ReflectionsWithLayerSelection-macOS
//
//  Created by Sugu on 2021/01/05.
//  Copyright © 2021 Apple. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#import "AAPLShaderTypes.h"

#define MAX_ITERATION 2000
#define STEP_SIZE 2
#define MAX_THICKNESS 0.001

// Compute the position and the reflection direction of the sample in texture space.
void ComputePosAndReflection(uint2 tid,
                             const constant SceneInfo& sceneInfo,
                             float3 vSampleNormalInVS,
                             texture2d<float, access::sample> tex_depth,
                             thread float3& outSamplePosInTS,
                             thread float3& outReflDirInTS,
                             thread float& outMaxDistance)
{
    float sampleDepth = tex_depth.read(tid).x;
    float4 samplePosInCS =  float4((float2(tid)/sceneInfo.ViewSize)*2-1.0f, sampleDepth, 1);
    samplePosInCS.y *= -1;

    float4 samplePosInVS = sceneInfo.InvProjMat * samplePosInCS;
    samplePosInVS /= samplePosInVS.w;

    float3 vCamToSampleInVS = normalize(samplePosInVS.xyz);
    float4 vReflectionInVS = float4(reflect(vCamToSampleInVS.xyz, vSampleNormalInVS.xyz),0);

    float4 vReflectionEndPosInVS = samplePosInVS + vReflectionInVS * 1000;
    float4 vReflectionEndPosInCS = sceneInfo.ProjMat * float4(vReflectionEndPosInVS.xyz, 1);
    vReflectionEndPosInCS /= vReflectionEndPosInCS.w;
    float3 vReflectionDir = normalize((vReflectionEndPosInCS - samplePosInCS).xyz);

    // Transform to texture space
    samplePosInCS.xy *= float2(0.5f, -0.5f);
    samplePosInCS.xy += float2(0.5f, 0.5f);
    
    vReflectionDir.xy *= float2(0.5f, -0.5f);
    
    outSamplePosInTS = samplePosInCS.xyz;
    outReflDirInTS = vReflectionDir;
    
	// Compute the maximum distance to trace before the ray goes outside of the visible area.
    outMaxDistance = outReflDirInTS.x>=0 ? (1 - outSamplePosInTS.x)/outReflDirInTS.x  : -outSamplePosInTS.x/outReflDirInTS.x;
    outMaxDistance = min(outMaxDistance, outReflDirInTS.y<0 ? (-outSamplePosInTS.y/outReflDirInTS.y)  : ((1-outSamplePosInTS.y)/outReflDirInTS.y));
    outMaxDistance = min(outMaxDistance, (1-outSamplePosInTS.z)/outReflDirInTS.z);
}

bool FindIntersection_Linear(float3 samplePosInTS,
                             float3 vReflDirInTS,
                             float maxTraceDistance,
                             texture2d<float, access::sample> tex_depth,
                             const constant SceneInfo& sceneInfo,
                             thread float3& intersection)
{
    constexpr sampler pointSampler;
    
    float3 vReflectionEndPosInTS = samplePosInTS + vReflDirInTS * maxTraceDistance;
    
    float3 dp = vReflectionEndPosInTS.xyz - samplePosInTS.xyz;
    int2 sampleScreenPos = int2(samplePosInTS.xy * sceneInfo.ViewSize.xy);
    int2 endPosScreenPos = int2(vReflectionEndPosInTS.xy * sceneInfo.ViewSize.xy);
    int2 dp2 = endPosScreenPos - sampleScreenPos;
    const int max_dist = max(abs(dp2.x), abs(dp2.y));
    dp /= max_dist;
    
    float4 rayPosInTS = float4(samplePosInTS.xyz + dp, 0);
    float4 vRayDirInTS = float4(dp.xyz, 0);
	float4 rayStartPos = rayPosInTS;

    int32_t hitIndex = -1;
    for(int i = 0;i<max_dist && i<MAX_ITERATION;i += 4)
    {
        float depth0 = 0;
        float depth1 = 0;
        float depth2 = 0;
        float depth3 = 0;

        float4 rayPosInTS0 = rayPosInTS+vRayDirInTS*0;
        float4 rayPosInTS1 = rayPosInTS+vRayDirInTS*1;
        float4 rayPosInTS2 = rayPosInTS+vRayDirInTS*2;
        float4 rayPosInTS3 = rayPosInTS+vRayDirInTS*3;

        depth3 = tex_depth.sample(pointSampler, rayPosInTS3.xy).x;
        depth2 = tex_depth.sample(pointSampler, rayPosInTS2.xy).x;
        depth1 = tex_depth.sample(pointSampler, rayPosInTS1.xy).x;
        depth0 = tex_depth.sample(pointSampler, rayPosInTS0.xy).x;

        {
            float thickness = rayPosInTS3.z - depth3;
            hitIndex = (thickness>=0 && thickness < MAX_THICKNESS) ? (i+3) : hitIndex;
        }
        {
            float thickness = rayPosInTS2.z - depth2;
            hitIndex = (thickness>=0 && thickness < MAX_THICKNESS) ? (i+2) : hitIndex;
        }
        {
            float thickness = rayPosInTS1.z - depth1;
            hitIndex = (thickness>=0 && thickness < MAX_THICKNESS) ? (i+1) : hitIndex;
        }
        {
            float thickness = rayPosInTS0.z - depth0;
            hitIndex = (thickness>=0 && thickness < MAX_THICKNESS) ? (i+0) : hitIndex;
        }

        if(hitIndex != -1) break;

        rayPosInTS = rayPosInTS3 + vRayDirInTS;
    }

    bool intersected = hitIndex >= 0;
    intersection = rayStartPos.xyz + vRayDirInTS.xyz * hitIndex;
     
    return intersected;
}

// hi-z with min-z buffer

float3 intersectDepthPlane(float3 o, float3 d, float z)
{
	return o + d * z;
}

float2 getCell(float2 pos, float2 cell_count)
{
	return float2(floor(pos*cell_count));
}

float3 intersectCellBoundary(float3 o, float3 d, float2 cell, float2 cell_count, float2 crossStep, float2 crossOffset, float min_z, float max_z)
{
	float3 intersection = 0;
	
	float2 index = cell + crossStep;
	index /= cell_count;
	index += crossOffset;
	
	float2 delta = index - o.xy;
	delta /= d.xy;
	float t = min(delta.x, delta.y);
	
	intersection = intersectDepthPlane(o, d, t);
	
	return intersection;
}

float getMinimumDepthPlane(float2 p, int mipLevel, texture2d<float, access::sample> tex_hi_z)
{
	constexpr sampler pointSampler(mip_filter::nearest);
	return tex_hi_z.sample(pointSampler, p, level(mipLevel)).x;
}

float2 getMinMaxDepthPlane(float2 p, int mipLevel, texture2d<float, access::sample> tex_hi_z)
{
	constexpr sampler pointSampler(mip_filter::nearest);
	return tex_hi_z.sample(pointSampler, p, level(mipLevel)).xy;
}

float2 getCellCount(int mipLevel, texture2d<float, access::sample> tex_hi_z)
{
	return float2(tex_hi_z.get_width(mipLevel) ,tex_hi_z.get_height(mipLevel));
}
 
bool crossedCellBoundary(float2 oldCellIndex, float2 newCellIndex)
{
	return (oldCellIndex.x != newCellIndex.x) || (oldCellIndex.y != newCellIndex.y);
}

bool FindIntersection_HiZ(float3 samplePosInTS,
                         float3 vReflDirInTS,
                          float maxTraceDistance,
                         texture2d<float, access::sample> tex_hi_z,
                         const constant SceneInfo& sceneInfo,
                         thread float3& intersection)
{
    float3 ray = samplePosInTS.xyz;
	float minZ = ray.z;
    float maxZ = samplePosInTS.z + vReflDirInTS.z * maxTraceDistance;
	float deltaZ = (maxZ - minZ);
    float3 d = vReflDirInTS * maxTraceDistance;
    
    float2 crossStep = float2(vReflDirInTS.x>=0 ? 1 : -1, vReflDirInTS.y>=0 ? 1 : -1);
	float2 crossOffset = crossStep  / sceneInfo.ViewSize / 128;
    crossStep = saturate(crossStep);
    
	int startLevel = 2;
	int stopLevel = 0;
	float2 startCellCount = getCellCount(startLevel, tex_hi_z);
	
	float3 o = ray;
    float2 rayCell = getCell(ray.xy, startCellCount);

    ray = intersectCellBoundary(o, d, rayCell, startCellCount, crossStep, crossOffset, minZ, maxZ);
    
    const int maxLevel = tex_hi_z.get_num_mip_levels()-1;
    
    int level = startLevel;
    int iter = 0;
    while(level >=stopLevel && ray.z <= maxZ && iter<MAX_ITERATION)
    {
        const float2 cellCount = getCellCount(level, tex_hi_z);
        const float2 oldCellIdx = getCell(ray.xy, cellCount);
		
		float cell_minZ = getMinimumDepthPlane((oldCellIdx+0.5f)/cellCount, level, tex_hi_z);
        float3 tmpRay = cell_minZ > ray.z ? intersectDepthPlane(o, d, (cell_minZ - minZ)/deltaZ) : ray;
        
        const float2 newCellIdx = getCell(tmpRay.xy, cellCount);
        
		float thickness = level == 0 ? (ray.z - cell_minZ) : 0;
        bool crossed = thickness>MAX_THICKNESS || crossedCellBoundary(oldCellIdx, newCellIdx);
        ray = crossed ? intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep, crossOffset, minZ, maxZ) : tmpRay;
        level = crossed ? min((float)maxLevel, level + 1.0f) : level-1;
		
        iter = iter + 1;
    }
    
    bool intersected = level < stopLevel && ray.y < 1;
    intersection = ray;
    
    return intersected;
}

float4 ComputeReflectedColor(float3 intersection,
                             texture2d<float, access::sample> tex_scene_color)
{
    constexpr sampler pointSampler(mip_filter::nearest);
    return float4(tex_scene_color.sample(pointSampler, intersection.xy));
}

kernel void kernel_screen_space_reflection_linear(texture2d<float, access::sample> tex_normal_refl_mask [[texture(0)]],
                                                  texture2d<float, access::sample> tex_depth [[texture(1)]],
                                                  texture2d<float, access::sample> tex_scene_color [[texture(2)]],
                                                  texture2d<float, access::write> tex_output [[texture(3)]],
                                                  const constant SceneInfo& sceneInfo [[buffer(0)]],
                                                  uint2 tid [[thread_position_in_grid]])
{
    float4 finalColor = 0;

    float4 NormalAndReflectionMask = tex_normal_refl_mask.read(tid);
    float4 color = tex_scene_color.read(tid);
	float4 normalInWS = float4(normalize(NormalAndReflectionMask.xyz), 0);
	float3 normal = (sceneInfo.ViewMat * normalInWS).xyz;
    float reflection_mask = NormalAndReflectionMask.w;

    float4 reflectionColor = 0;
    if(reflection_mask != 0)
    {
        float3 samplePosInTS = 0;
        float3 vReflDirInTS = 0;
        float maxTraceDistance = 0;

        // Compute the position and the reflection vector of this sample in texture space.
        ComputePosAndReflection(tid, sceneInfo, normal, tex_depth, samplePosInTS, vReflDirInTS, maxTraceDistance);

        // Find intersection in texture space by tracing the reflection ray
        float3 intersection = 0;
		if(vReflDirInTS.z>0.0)
		{
			bool intersected = FindIntersection_Linear(samplePosInTS, vReflDirInTS, maxTraceDistance, tex_depth, sceneInfo, intersection);
			
			// Compute reflection color if intersected
			reflectionColor = intersected ? ComputeReflectedColor(intersection, tex_scene_color) : 0;
		}
    }

    // Add the reflection color to the color of the sample.
    finalColor = color + reflectionColor;

    tex_output.write(finalColor, tid);
}


kernel void kernel_screen_space_reflection_hi_z(texture2d<float, access::sample> tex_normal_refl_mask [[texture(0)]],
                                                texture2d<float, access::sample> tex_hi_z [[texture(1)]],
                                                texture2d<float, access::sample> tex_scene_color [[texture(2)]],
                                                texture2d<float, access::write> tex_output [[texture(3)]],
                                                const constant SceneInfo& sceneInfo [[buffer(0)]],
                                                uint2 tid [[thread_position_in_grid]])
{
    float4 finalColor = 0;
    
    float4 NormalAndReflectionMask = tex_normal_refl_mask.read(tid);
    float4 color = tex_scene_color.read(tid);
	float4 normalInWS = float4(normalize(NormalAndReflectionMask.xyz), 0);
	float3 normal = (sceneInfo.ViewMat * normalInWS).xyz;
    float reflection_mask = NormalAndReflectionMask.w;
    
    float4 reflectionColor = 0;
    if(reflection_mask != 0)
    {
        float3 samplePosInTS = 0;
        float3 vReflDirInTS = 0;
        float maxTraceDistance = 0;
        // Compute the position and the reflection vector of this sample in texture space.
        ComputePosAndReflection(tid, sceneInfo, normal, tex_hi_z, samplePosInTS, vReflDirInTS, maxTraceDistance);
        
        // Find intersection in texture space by tracing the reflection ray
        float3 intersection = 0;
        if(vReflDirInTS.z>0.0)
        {
            bool intersected = FindIntersection_HiZ(samplePosInTS, vReflDirInTS, maxTraceDistance, tex_hi_z, sceneInfo, intersection);
            
            // Compute reflection color if intersected
            reflectionColor = intersected ? ComputeReflectedColor(intersection, tex_scene_color) : 0;
        }
    }
    
    // Add the reflection color to the color of the sample.
    finalColor = color + reflectionColor;
    
    tex_output.write(finalColor, tid);
}

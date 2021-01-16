/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of renderer class which performs Metal setup and per frame rendering
*/

#import <ModelIO/ModelIO.h>
#import <MetalKit/MetalKit.h>
#import <vector>

#import "AAPLRenderer.h"
#import "AAPLMesh.h"
#import "AAPLMathUtilities.h"

// Include header shared between C code here, which executes Metal API commands,
// and .metal files
#import "AAPLShaderTypes.h"

#include "AAPLRendererUtils.h"

static const NSUInteger    MaxBuffersInFlight       = 3;  // Number of in-flight command buffers
static const NSUInteger    MaxActors                = 32; // Max possible actors
static const NSUInteger    MaxVisibleFaces          = 5;  // Number of faces an actor could be visible in
static const NSUInteger    CubemapResolution        = 256;
static const vector_float3 SceneCenter              = (vector_float3){0.f, -250.f, 1000.f};
vector_float3 CameraDistanceFromCenter = (vector_float3){0.f, 10.f, -550.f};
static const vector_float3 CameraRotationAxis       = (vector_float3){0,1,0};
float anim_speed_factor = 0.5f;
static const float         CameraRotationSpeed      = 0.0025f;
static const float         ActorRotationSpeed       = 1;

#define SSR_HI_Z 1	

// Main class performing the rendering
@implementation AAPLRenderer
{
    dispatch_semaphore_t _inFlightSemaphore;
    id<MTLDevice>       _device;
    id<MTLCommandQueue> _commandQueue;

    // Current frame number modulo MaxBuffersInFlight.
    // Tells which buffer index is writable in buffer arrays.
    uint8_t _uniformBufferIndex;

    // CPU app-specific data
    Camera                           _cameraFinal;
    NSMutableArray <AAPLActorData*>* _actorData;

    // Dynamic GPU buffers
    id<MTLBuffer> _frameParamsBuffers                [MaxBuffersInFlight]; // frame-constant parameters
    id<MTLBuffer> _viewportsParamsBuffers_final      [MaxBuffersInFlight]; // frame-constant parameters, final viewport
    id<MTLBuffer> _actorsParamsBuffers               [MaxBuffersInFlight]; // per-actor parameters
    id<MTLBuffer> _instanceParamsBuffers_final       [MaxBuffersInFlight]; // per-instance parameters for final pass
    
    id<MTLBuffer> _sceneInfoBuffers[MaxBuffersInFlight];

    id<MTLDepthStencilState> _depthState;
    
    id<MTLTexture> _gbuffer_diffuse;
    id<MTLTexture> _gbuffer_normal_refl_mask;
    id<MTLTexture> _gbuffer_depth;
    id<MTLTexture> _ssr_rayDir;
    id<MTLTexture> _directionalLighting_output;
    id<MTLTexture> _final_output;
    id<MTLTexture> _depth_hi_z;
    std::vector<id<MTLTexture>> _depth_hi_z_levels;
    
    //
    id<MTLComputePipelineState> _pipeline_lighting;
	
    id<MTLComputePipelineState> _pipeline_ssr_linear;
    id<MTLComputePipelineState> _pipeline_ssr_hi_z;

    id<MTLComputePipelineState> _createHiZ;
    id<MTLComputePipelineState> _depthCopy;

    id<MTLRenderPipelineState> _pipeline_copy;
    
    id<MTLFence> _gbuffer_fence;

    vector_float2 _viewSize;
	
	bool _AnimationEnabled;
	bool _SSREnabled;
	bool _SSRTechinque; // true : linear false : hi-z
}
-(void)onToggleSSRButton:(NSButton *)button
{
	_SSREnabled = !_SSREnabled;
	
	[button setTitle:_SSREnabled ? @"SSR Enabled" : @"SSR Disabled"];
	
}
-(void)onToggleSSRTechniqueButton:(NSButton *)button
{
	_SSRTechinque = !_SSRTechinque;
	
	[button setTitle:_SSRTechinque ? @"Linear Tracing" : @"Hi-Z Tracing"];
}
-(void)onMoveCamPosSlider:(NSSlider *)slider
{
	float ratio = slider.floatValue / 100.0f;
	float camPosMin = 10;
	float camPosMax = 400;
	float camPos = camPosMin + (camPosMax - camPosMin) * ratio;
	
	CameraDistanceFromCenter.y = camPos;
}
-(void)onAnimSpeedSlider:(NSSlider *)slider
{
	float ratio = slider.floatValue / 100.0f;
	float speed_min = 0;
	float speed_max = 2;
	
	anim_speed_factor = speed_min + (speed_max - speed_min) * ratio;
}
- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
{
    self = [super init];
    if(self)
    {
        _device = mtkView.device;
        _inFlightSemaphore = dispatch_semaphore_create(MaxBuffersInFlight);
        [self loadMetalWithMetalKitView:mtkView];
        [self loadAssetsWithMetalKitView:mtkView];
		
		_SSREnabled = false;
		_SSRTechinque = true;
		_AnimationEnabled = false;
    }

    return self;
}

- (void)loadComputeKernels
{
    NSError* error = nil;
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    
    {
        id<MTLFunction> kernel_func = [defaultLibrary newFunctionWithName:@"kernel_lighting"];
        if (kernel_func == nil)
        {
            NSLog(@"Failed to find the function.");
            return;
        }

        _pipeline_lighting = [_device newComputePipelineStateWithFunction: kernel_func error:&error];
        if (_pipeline_lighting == nil)
        {
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return;
        }
    }
    
    {
        id<MTLFunction> kernel_func = [defaultLibrary newFunctionWithName:@"kernel_screen_space_reflection_linear"];
        if (kernel_func == nil)
        {
            NSLog(@"Failed to find the function.");
            return;
        }

        _pipeline_ssr_linear = [_device newComputePipelineStateWithFunction: kernel_func error:&error];
        if (_pipeline_ssr_linear == nil)
        {
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return;
        }
    }
    
    {
        id<MTLFunction> kernel_func = [defaultLibrary newFunctionWithName:@"kernel_screen_space_reflection_hi_z"];
        if (kernel_func == nil)
        {
            NSLog(@"Failed to find the function.");
            return;
        }

        _pipeline_ssr_hi_z = [_device newComputePipelineStateWithFunction: kernel_func error:&error];
        if (_pipeline_ssr_hi_z == nil)
        {
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return;
        }
    }
    
    {
        id<MTLFunction> kernel_func = [defaultLibrary newFunctionWithName:@"kernel_createHiZ"];
        if (kernel_func == nil)
        {
            NSLog(@"Failed to find the function.");
            return;
        }

        _createHiZ = [_device newComputePipelineStateWithFunction: kernel_func error:&error];
        if (_createHiZ == nil)
        {
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return;
        }
    }
    
    {
        id<MTLFunction> kernel_func = [defaultLibrary newFunctionWithName:@"kernel_depthCopy"];
        if (kernel_func == nil)
        {
            NSLog(@"Failed to find the function.");
            return;
        }

        _depthCopy = [_device newComputePipelineStateWithFunction: kernel_func error:&error];
        if (_depthCopy == nil)
        {
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return;
        }
    }

}
- (void)createGBuffer:(nonnull MTKView *)mtkView
{
    uint32_t width = mtkView.drawableSize.width;
    uint32_t height = mtkView.drawableSize.height;
    
    _viewSize.x = width;
    _viewSize.y = height;
    
    {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:width height:height mipmapped:false];
    
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        _gbuffer_normal_refl_mask = [_device newTextureWithDescriptor:desc];
    }
    {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:width height:height mipmapped:false];
    
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        _ssr_rayDir = [_device newTextureWithDescriptor:desc];
    }

    {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:false];
    
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        _gbuffer_diffuse = [_device newTextureWithDescriptor:desc];
    }
    {
        MTLPixelFormat depthFormat = MTLPixelFormatDepth32Float;

        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:depthFormat width:width height:height mipmapped:false];

        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        
        _gbuffer_depth = [_device newTextureWithDescriptor:desc];
    }
    {
        MTLPixelFormat pixelFormat = MTLPixelFormatR32Float;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat width:width height:height mipmapped:true];

        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        _depth_hi_z = [_device newTextureWithDescriptor:desc];
        
        _depth_hi_z_levels.clear();
        for(size_t mipLevel = 0;mipLevel<_depth_hi_z.mipmapLevelCount;++mipLevel)
        {
            _depth_hi_z_levels.push_back([_depth_hi_z newTextureViewWithPixelFormat:pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(mipLevel, 1) slices:NSMakeRange(0, 1)]);
        }
    }
    {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:width height:height mipmapped:false];
    
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
        
        _directionalLighting_output = [_device newTextureWithDescriptor:desc];
        _final_output = [_device newTextureWithDescriptor:desc];
    }
    
    _gbuffer_fence = [_device newFence];
    
}
- (void)loadMetalWithMetalKitView:(nonnull MTKView *)mtkView
{
    // Create and load our basic Metal state objects

    // We allocate MaxBuffersInFlight instances of uniform buffers. This allows us to
    // update uniforms as a ring (i.e. triple buffer the uniform data) so that the GPU
    // reads from one slot in the ring while the CPU writes to another.

    for (int i = 0; i < MaxBuffersInFlight; i++)
    {
        id<MTLBuffer> frameParamsBuffer =
            [_device newBufferWithLength: sizeof(FrameParams)
                                 options: MTLResourceStorageModeShared];
        frameParamsBuffer.label = [NSString stringWithFormat:@"frameParams[%i]", i];
        _frameParamsBuffers[i] = frameParamsBuffer;

        id<MTLBuffer> finalViewportParamsBuffer =
            [_device newBufferWithLength: sizeof(ViewportParams)
                                 options: MTLResourceStorageModeShared];
        finalViewportParamsBuffer .label = [NSString stringWithFormat:@"viewportParams_final[%i]", i];
        _viewportsParamsBuffers_final[i] = finalViewportParamsBuffer;

        id<MTLBuffer> sceneInfoBuffer =
            [_device newBufferWithLength: sizeof(SceneInfo)
                                 options: MTLResourceStorageModeShared];
        sceneInfoBuffer.label = [NSString stringWithFormat:@"sceneInfo[%i]", i];
        _sceneInfoBuffers[i] = sceneInfoBuffer;

        
        // This buffer will contain every actor's data required by shaders.
        //
        // When rendering a batch (aka an actor), the shader will access its actor data
        //   through a reference, without knowing about the actual offset of the data within
        //   the buffer.
        // This is done by, before each draw call, setting the buffer in the Metal framework with
        //   an explicit offset when setting the buffer
        //
        // As this offset _has_ to be 256 bytes aligned, that means we'll need to round up
        // the size of an ActorData to the next multiple of 256.
        id<MTLBuffer> actorParamsBuffer =
            [_device newBufferWithLength: Align<BufferOffsetAlign> (sizeof(ActorParams)) * MaxActors
                                 options: MTLResourceStorageModeShared];
        actorParamsBuffer.label = [NSString stringWithFormat:@"actorsParams[%i]", i];
        _actorsParamsBuffers[i] = actorParamsBuffer;

        // No need to align these, as the shader will be provided a pointer to the buffer's
        //   beginning, and index into it, like an array, in the shader code itself
        id<MTLBuffer> finalInstanceParamsBuffer =
            [_device newBufferWithLength: MaxActors*sizeof(InstanceParams)
                                 options: MTLResourceStorageModeShared];
        finalInstanceParamsBuffer.label = [NSString stringWithFormat:@"instanceParams_final[%i]", i];

        // There is only one viewport in the final pass, which is at viewportIndex 0.  So set every
        //   viewportIndex for each actor's final pass to 0
        for(NSUInteger actorIdx = 0; actorIdx < MaxActors; actorIdx++)
        {
            InstanceParams *instanceParams =
                ((InstanceParams*)finalInstanceParamsBuffer.contents)+actorIdx;
            instanceParams->viewportIndex = 0;
        }
        _instanceParamsBuffers_final[i] = finalInstanceParamsBuffer;

    }

    mtkView.sampleCount               = 1;
    mtkView.colorPixelFormat          = MTLPixelFormatBGRA8Unorm_sRGB;

    MTLDepthStencilDescriptor *depthStateDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStateDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthStateDesc.depthWriteEnabled    = YES;

    _depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];

    // Create the command queue
    _commandQueue = [_device newCommandQueue];
    
    [self createGBuffer:mtkView];
    [self loadComputeKernels];
}

- (void)loadAssetsWithMetalKitView:(nonnull MTKView*)mtkView
{
    //-------------------------------------------------------------------------------------------
    // Create a vertex descriptor for our Metal pipeline. Specifies the layout of vertices the
    //   pipeline should expect.  The layout below keeps attributes used to calculate vertex shader
    //   output (world position, skinning, tweening weights...) separate from other
    //   attributes (texture coordinates, normals).  This generally maximizes pipeline efficiency.

    MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init];

    // Positions.
    mtlVertexDescriptor.attributes[VertexAttributePosition].format       = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttributePosition].offset       = 0;
    mtlVertexDescriptor.attributes[VertexAttributePosition].bufferIndex  = BufferIndexMeshPositions;

    // Texture coordinates.
    mtlVertexDescriptor.attributes[VertexAttributeTexcoord].format       = MTLVertexFormatFloat2;
    mtlVertexDescriptor.attributes[VertexAttributeTexcoord].offset       = 0;
    mtlVertexDescriptor.attributes[VertexAttributeTexcoord].bufferIndex  = BufferIndexMeshGenerics;

    // Normals.
    mtlVertexDescriptor.attributes[VertexAttributeNormal].format         = MTLVertexFormatHalf4;
    mtlVertexDescriptor.attributes[VertexAttributeNormal].offset         = 8;
    mtlVertexDescriptor.attributes[VertexAttributeNormal].bufferIndex    = BufferIndexMeshGenerics;

    // Tangents
    mtlVertexDescriptor.attributes[VertexAttributeTangent].format        = MTLVertexFormatHalf4;
    mtlVertexDescriptor.attributes[VertexAttributeTangent].offset        = 16;
    mtlVertexDescriptor.attributes[VertexAttributeTangent].bufferIndex   = BufferIndexMeshGenerics;

    // Bitangents
    mtlVertexDescriptor.attributes[VertexAttributeBitangent].format      = MTLVertexFormatHalf4;
    mtlVertexDescriptor.attributes[VertexAttributeBitangent].offset      = 24;
    mtlVertexDescriptor.attributes[VertexAttributeBitangent].bufferIndex = BufferIndexMeshGenerics;

    // Position Buffer Layout
    mtlVertexDescriptor.layouts[BufferIndexMeshPositions].stride         = 12;
    mtlVertexDescriptor.layouts[BufferIndexMeshPositions].stepRate       = 1;
    mtlVertexDescriptor.layouts[BufferIndexMeshPositions].stepFunction   = MTLVertexStepFunctionPerVertex;

    // Generic Attribute Buffer Layout
    mtlVertexDescriptor.layouts[BufferIndexMeshGenerics].stride          = 32;
    mtlVertexDescriptor.layouts[BufferIndexMeshGenerics].stepRate        = 1;
    mtlVertexDescriptor.layouts[BufferIndexMeshGenerics].stepFunction    = MTLVertexStepFunctionPerVertex;

    //-------------------------------------------------------------------------------------------

    NSError *error = NULL;

    // Load all the shader files with a metal file extension in the project
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];

    // Create a reusable pipeline state
    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.vertexDescriptor                = mtlVertexDescriptor;
    pipelineStateDescriptor.inputPrimitiveTopology          = MTLPrimitiveTopologyClassTriangle;
    pipelineStateDescriptor.vertexFunction =
        [defaultLibrary newFunctionWithName:@"vertexTransform"];
    pipelineStateDescriptor.fragmentFunction =
        [defaultLibrary newFunctionWithName:@"fragmentGBuffer"];
    pipelineStateDescriptor.sampleCount                     = mtkView.sampleCount;
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    pipelineStateDescriptor.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA8Unorm;
    pipelineStateDescriptor.depthAttachmentPixelFormat      = MTLPixelFormatDepth32Float;

    pipelineStateDescriptor.label = @"TemplePipeline";
    id<MTLRenderPipelineState> templePipelineState =
        [_device newRenderPipelineStateWithDescriptor: pipelineStateDescriptor error:&error];
    
    NSAssert(templePipelineState, @"Failed to create pipeline state: %@", error);

    pipelineStateDescriptor.label = @"GroundPipeline";
    pipelineStateDescriptor.fragmentFunction =
        [defaultLibrary newFunctionWithName:@"fragmentGroundGBuffer"];
    id<MTLRenderPipelineState> groundPipelineState  =
        [_device newRenderPipelineStateWithDescriptor: pipelineStateDescriptor error:&error];
    
    NSAssert(groundPipelineState, @"Failed to create pipeline state: %@", error);

    pipelineStateDescriptor.label = @"WallPipeline";
    pipelineStateDescriptor.fragmentFunction =
        [defaultLibrary newFunctionWithName:@"fragmentWallGBuffer"];
    id<MTLRenderPipelineState> wallPipelineState  =
        [_device newRenderPipelineStateWithDescriptor: pipelineStateDescriptor error:&error];
    
    NSAssert(groundPipelineState, @"Failed to create pipeline state: %@", error);

    
    _cameraFinal.rotation = 0;

    // Create a Model I/O vertexDescriptor so that we format/layout our Model I/O mesh vertices to
    //   fit our Metal render pipeline's vertex descriptor layout
    MDLVertexDescriptor *modelIOVertexDescriptor =
    MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor);

    // Indicate how each Metal vertex descriptor attribute maps to each Model I/O  attribute
    modelIOVertexDescriptor.attributes[VertexAttributePosition].name  = MDLVertexAttributePosition;
    modelIOVertexDescriptor.attributes[VertexAttributeTexcoord].name  = MDLVertexAttributeTextureCoordinate;
    modelIOVertexDescriptor.attributes[VertexAttributeNormal].name    = MDLVertexAttributeNormal;
    modelIOVertexDescriptor.attributes[VertexAttributeTangent].name   = MDLVertexAttributeTangent;
    modelIOVertexDescriptor.attributes[VertexAttributeBitangent].name = MDLVertexAttributeBitangent;

    NSURL *modelFileURL = [[NSBundle mainBundle] URLForResource: @"Models/Temple.obj"
                                                  withExtension:  nil];

    NSAssert(modelFileURL,
             @"Could not find model file (%@) in bundle",
             modelFileURL.absoluteString);

    MDLAxisAlignedBoundingBox templeAabb;
    NSArray <AAPLMesh*>* templeMeshes = [AAPLMesh newMeshesFromUrl: modelFileURL
                                           modelIOVertexDescriptor: modelIOVertexDescriptor
                                                       metalDevice: _device
                                                             error: &error
                                                              aabb: templeAabb];
    
    NSAssert(templeMeshes, @"Could not create meshes from model file: %@", modelFileURL.absoluteString);
    
    vector_float4 templeBSphere;
    templeBSphere.xyz = (templeAabb.maxBounds + templeAabb.minBounds)*0.5;
    templeBSphere.w = vector_length ((templeAabb.maxBounds - templeAabb.minBounds)*0.5);


    MTKMeshBufferAllocator *meshBufferAllocator =
        [[MTKMeshBufferAllocator alloc] initWithDevice:_device];

    MDLMesh* mdlSphere = [MDLMesh newEllipsoidWithRadii: 100.0
                                         radialSegments: 30
                                       verticalSegments: 20
                                           geometryType: MDLGeometryTypeTriangles
                                          inwardNormals: false
                                             hemisphere: false
                                              allocator: meshBufferAllocator];

    vector_float4 sphereBSphere;
    sphereBSphere.xyz = (vector_float3){0,0,0};
    sphereBSphere.w = 200.f;

    NSArray <AAPLMesh*>* sphereMeshes = [AAPLMesh newMeshesFromObject: mdlSphere
                                              modelIOVertexDescriptor: modelIOVertexDescriptor
                                                metalKitTextureLoader: NULL
                                                          metalDevice: _device
                                                                error: &error ];

    NSAssert(sphereMeshes, @"Could not create sphere meshes: %@", error);

    MDLMesh* mdlGround = [MDLMesh newPlaneWithDimensions: {100000.f, 100000.f}
                                                segments: {1,1}
                                            geometryType: MDLGeometryTypeTriangles
                                               allocator: meshBufferAllocator];

    vector_float4 groundBSphere;
    groundBSphere.xyz = (vector_float3){0,0,0};
    groundBSphere.w = 1415.f;

    NSArray <AAPLMesh*>* groundMeshes = [AAPLMesh newMeshesFromObject: mdlGround
                                              modelIOVertexDescriptor: modelIOVertexDescriptor
                                                metalKitTextureLoader: NULL
                                                          metalDevice: _device
                                                                error: &error ];
    
    
    MDLMesh* mdlWall = [MDLMesh newBoxWithDimensions:{100.f,300.f,300.f} segments:{1,1,1} geometryType:MDLGeometryTypeTriangles inwardNormals:false allocator:meshBufferAllocator];

    vector_float4 wallBSphere;
    wallBSphere.xyz = (vector_float3){0,0,0};
    wallBSphere.w = 1415.f;

    NSArray <AAPLMesh*>* wallMeshes = [AAPLMesh newMeshesFromObject: mdlWall
                                              modelIOVertexDescriptor: modelIOVertexDescriptor
                                                metalKitTextureLoader: NULL
                                                          metalDevice: _device
                                                                error: &error ];
	
	MDLMesh* mdlTallWall = [MDLMesh newBoxWithDimensions:{100.f,3000.f,300.f} segments:{1,1,1} geometryType:MDLGeometryTypeTriangles inwardNormals:false allocator:meshBufferAllocator];

	vector_float4 tallWallBSphere;
	tallWallBSphere.xyz = (vector_float3){0,0,0};
	tallWallBSphere.w = 1415.f;

	NSArray <AAPLMesh*>* tallWallMeshes = [AAPLMesh newMeshesFromObject: mdlTallWall
											  modelIOVertexDescriptor: modelIOVertexDescriptor
												metalKitTextureLoader: NULL
														  metalDevice: _device
																error: &error ];
    
    NSAssert(groundMeshes, @"Could not create ground meshes: %@", error);

    // Finally, we create the actor list :
    _actorData = [NSMutableArray new];
    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3) {-1000, -150.f, 1000.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 1.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {1.f, 1.f, 1.f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;

    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3) {1000.f, -150.f, 1000.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 2.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {0.6f, 1.f, 0.6f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;

    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3) {1150.f, -150.f, -400.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 3.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {0.45f, 0.45f, 1.f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;

    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3) {-1200.f, -150.f, -300.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 4.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {1.f, 0.6f, 0.6f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;

    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3){0.f, -155.f, 0.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 0.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {1.f, 1.f, 1.f};
    _actorData.lastObject.bSphere           = groundBSphere;
    _actorData.lastObject.gpuProg           = groundPipelineState;
    _actorData.lastObject.meshes            = groundMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;
    
    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3){200.f, -120, 200.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 5.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {1.f, 1.f, 1.f};
    _actorData.lastObject.bSphere           = wallBSphere;
    _actorData.lastObject.gpuProg           = wallPipelineState;
    _actorData.lastObject.meshes            = sphereMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;
	
	[_actorData addObject:[AAPLActorData new]];
	_actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
	_actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3){0.f, 0, 3000.f};
	_actorData.lastObject.rotationAmount    = 0.f;
	_actorData.lastObject.rotationSpeed     = 2.f;
	_actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
	_actorData.lastObject.diffuseMultiplier = (vector_float3) {0.f, 2.f, 2.f};
	_actorData.lastObject.bSphere           = tallWallBSphere;
	_actorData.lastObject.gpuProg           = templePipelineState;
	_actorData.lastObject.meshes            = tallWallMeshes;
	_actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;
	
	[_actorData addObject:[AAPLActorData new]];
	_actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
	_actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3){2500.f, 0, -2500.f};
	_actorData.lastObject.rotationAmount    = 0.f;
	_actorData.lastObject.rotationSpeed     = 2.f;
	_actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
	_actorData.lastObject.diffuseMultiplier = (vector_float3) {2.f, 2.f, 0.f};
	_actorData.lastObject.bSphere           = tallWallBSphere;
	_actorData.lastObject.gpuProg           = templePipelineState;
	_actorData.lastObject.meshes            = tallWallMeshes;
	_actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;
	
	[_actorData addObject:[AAPLActorData new]];
	_actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
	_actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3){2500.f, 0, 2500.f};
	_actorData.lastObject.rotationAmount    = 0.f;
	_actorData.lastObject.rotationSpeed     = 1.f;
	_actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
	_actorData.lastObject.diffuseMultiplier = (vector_float3) {0.f, 2.f, 0.f};
	_actorData.lastObject.bSphere           = tallWallBSphere;
	_actorData.lastObject.gpuProg           = templePipelineState;
	_actorData.lastObject.meshes            = tallWallMeshes;
	_actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;
	
	[_actorData addObject:[AAPLActorData new]];
	_actorData.lastObject.translation       = (vector_float3) {0.f, 300.f, 0.f};
	_actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3){-3500.f, 0, 3500.f};
	_actorData.lastObject.rotationAmount    = 0.f;
	_actorData.lastObject.rotationSpeed     = 1.f;
	_actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
	_actorData.lastObject.diffuseMultiplier = (vector_float3) {2.f, 0.f, 0.f};
	_actorData.lastObject.bSphere           = tallWallBSphere;
	_actorData.lastObject.gpuProg           = templePipelineState;
	_actorData.lastObject.meshes            = tallWallMeshes;
	_actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;


    {
        MTLRenderPipelineDescriptor *renderDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        renderDescriptor.sampleCount = 1;
        renderDescriptor.vertexFunction = [defaultLibrary newFunctionWithName:@"copyVertex"];
        renderDescriptor.fragmentFunction = [defaultLibrary newFunctionWithName:@"copyFragment"];
        renderDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat;

        _pipeline_copy = [_device newRenderPipelineStateWithDescriptor:renderDescriptor error:&error];
        
        if (!_pipeline_copy)
            NSLog(@"Failed to create pipeline state, error %@", error);
    }
}

- (void)updateGameState
{
    FrustumCuller culler_final;
    FrustumCuller culler_probe [6];

    // Update each actor's position and parameter buffer
    {
        ActorParams *actorParams  =
            (ActorParams *)_actorsParamsBuffers[_uniformBufferIndex].contents;

        for (int i = 0; i < _actorData.count; i++)
        {
            const matrix_float4x4 modelTransMatrix    = matrix4x4_translation(_actorData[i].translation);
            const matrix_float4x4 modelRotationMatrix = matrix4x4_rotation (_actorData[i].rotationAmount, _actorData[i].rotationAxis);
            const matrix_float4x4 modelPositionMatrix = matrix4x4_translation(_actorData[i].rotationPoint);

            matrix_float4x4 modelMatrix;
            modelMatrix = matrix_multiply(modelRotationMatrix, modelTransMatrix);
            modelMatrix = matrix_multiply(modelPositionMatrix, modelMatrix);

            _actorData[i].modelPosition = matrix_multiply(modelMatrix, (vector_float4) {0, 0, 0, 1});

            // we update the actor's rotation for next frame (cpu side) :
            _actorData[i].rotationAmount += 0.004 * _actorData[i].rotationSpeed * ActorRotationSpeed * anim_speed_factor;

            // we update the actor's shader parameters :
            actorParams[i].modelMatrix = modelMatrix;
            actorParams[i].diffuseMultiplier = _actorData[i].diffuseMultiplier;
            actorParams[i].materialShininess = 4;
        }
    }
    // We update the final viewport (shader parameter buffer + culling utility) :
    {
        _cameraFinal.target   = SceneCenter;

        _cameraFinal.rotation = fmod ((_cameraFinal.rotation + CameraRotationSpeed * anim_speed_factor), M_PI*2.f);
        matrix_float3x3 rotationMatrix = matrix3x3_rotation (_cameraFinal.rotation,  CameraRotationAxis);

        _cameraFinal.position = SceneCenter;
        _cameraFinal.position += matrix_multiply (rotationMatrix, CameraDistanceFromCenter);

        const matrix_float4x4 viewMatrix       = _cameraFinal.GetViewMatrix();
        const matrix_float4x4 projectionMatrix = _cameraFinal.GetProjectionMatrix_LH();

        culler_final.Reset_LH (viewMatrix, _cameraFinal);

        ViewportParams *viewportBuffer = (ViewportParams *)_viewportsParamsBuffers_final[_uniformBufferIndex].contents;
        viewportBuffer[0].cameraPos            = _cameraFinal.position;
        viewportBuffer[0].viewMatrix = viewMatrix;
        viewportBuffer[0].viewProjectionMatrix = matrix_multiply (projectionMatrix, viewMatrix);
        
        SceneInfo* sceneInfo = (SceneInfo*)(_sceneInfoBuffers[_uniformBufferIndex].contents);
        sceneInfo->ViewSize = _viewSize;
        sceneInfo->ViewMat = viewMatrix;
        sceneInfo->ProjMat = projectionMatrix;
        sceneInfo->InvProjMat = matrix_inverse_transpose(matrix_transpose(projectionMatrix));
    }
    // We update the shader parameters - frame constants :
    {
        const vector_float3 ambientLightColor         = {0.2, 0.2, 0.2};
        const vector_float3 directionalLightColor     = {.75, .75, .75};
        const vector_float3 directionalLightDirection = vector_normalize((vector_float3){1.0, -1.0, 1.0});

        FrameParams *frameParams =
            (FrameParams *) _frameParamsBuffers[_uniformBufferIndex].contents;
        frameParams[0].ambientLightColor            = ambientLightColor;
        frameParams[0].directionalLightInvDirection = -directionalLightDirection;
        frameParams[0].directionalLightColor        = directionalLightColor;
    }
    //  Perform culling and determine how many instances we need to draw
    {

        for (int actorIdx = 0; actorIdx < _actorData.count; actorIdx++)
        {
            if (_actorData[actorIdx].passFlags & EPassFlags::Final)
            {
                if (culler_final.Intersects (_actorData[actorIdx].modelPosition.xyz, _actorData[actorIdx].bSphere))
                {
                    _actorData[actorIdx].visibleInFinal = YES;
                }
                else
                {
                    _actorData[actorIdx].visibleInFinal = NO;
                }
            }
        }
    }
}

- (void)drawActors:(id<MTLRenderCommandEncoder>) renderEncoder
{
    id<MTLBuffer> viewportBuffer;
    id<MTLBuffer> visibleVpListPerActor;

    viewportBuffer        = _viewportsParamsBuffers_final [_uniformBufferIndex];
    visibleVpListPerActor = _instanceParamsBuffers_final  [_uniformBufferIndex];

    // Adds contextual info into the GPU Frame Capture tool
    [renderEncoder pushDebugGroup:[NSString stringWithFormat:@"DrawActors"]];

    [renderEncoder setCullMode:MTLCullModeBack];
    [renderEncoder setDepthStencilState:_depthState];

    // Set any buffers fed into our render pipeline

    [renderEncoder setFragmentBuffer: _frameParamsBuffers[_uniformBufferIndex]
                              offset: 0
                             atIndex: BufferIndexFrameParams];

    [renderEncoder setVertexBuffer: viewportBuffer
                            offset: 0
                           atIndex: BufferIndexViewportParams];

    [renderEncoder setFragmentBuffer: viewportBuffer
                              offset: 0
                             atIndex: BufferIndexViewportParams];

    [renderEncoder setVertexBuffer: visibleVpListPerActor
                            offset: 0
                           atIndex: BufferIndexInstanceParams];

    for (int actorIdx = 0; actorIdx < _actorData.count; actorIdx++)
    {
        AAPLActorData* lActor = _actorData[actorIdx];

        uint32_t visibleVpCount;

        visibleVpCount = lActor.visibleInFinal;

        if (visibleVpCount == 0) continue;

        // per-actor parameters
        [renderEncoder setVertexBuffer: _actorsParamsBuffers[_uniformBufferIndex]
                                offset: actorIdx * Align<BufferOffsetAlign> (sizeof(ActorParams))
                               atIndex: BufferIndexActorParams];

        [renderEncoder setFragmentBuffer: _actorsParamsBuffers[_uniformBufferIndex]
                                  offset: actorIdx * Align<BufferOffsetAlign> (sizeof(ActorParams))
                                 atIndex: BufferIndexActorParams];

        [renderEncoder setRenderPipelineState:lActor.gpuProg];

        for (AAPLMesh *mesh in lActor.meshes)
        {
            MTKMesh *metalKitMesh = mesh.metalKitMesh;

            // Set mesh's vertex buffers
            for (NSUInteger bufferIndex = 0; bufferIndex < metalKitMesh.vertexBuffers.count; bufferIndex++)
            {
                MTKMeshBuffer *vertexBuffer = metalKitMesh.vertexBuffers[bufferIndex];
                if((NSNull*)vertexBuffer != [NSNull null])
                {
                    [renderEncoder setVertexBuffer: vertexBuffer.buffer
                                            offset: vertexBuffer.offset
                                           atIndex: bufferIndex];
                }
            }

            // Draw each submesh of our mesh
            for(AAPLSubmesh *submesh in mesh.submeshes)
            {
                // Set any textures read/sampled from our render pipeline
                id<MTLTexture> tex;

                tex = submesh.textures [TextureIndexBaseColor];
                if ((NSNull*)tex != [NSNull null])
                {
                    [renderEncoder setFragmentTexture:tex atIndex:TextureIndexBaseColor];
                }

                tex = submesh.textures [TextureIndexNormal];
                if ((NSNull*)tex != [NSNull null])
                {
                    [renderEncoder setFragmentTexture:tex atIndex:TextureIndexNormal];
                }

                tex = submesh.textures[TextureIndexSpecular];
                if ((NSNull*)tex != [NSNull null])
                {
                    [renderEncoder setFragmentTexture:tex atIndex:TextureIndexSpecular];
                }

                MTKSubmesh *metalKitSubmesh = submesh.metalKitSubmmesh;

                [renderEncoder drawIndexedPrimitives: metalKitSubmesh.primitiveType
                                          indexCount: metalKitSubmesh.indexCount
                                           indexType: metalKitSubmesh.indexType
                                         indexBuffer: metalKitSubmesh.indexBuffer.buffer
                                   indexBufferOffset: metalKitSubmesh.indexBuffer.offset
                                       instanceCount: visibleVpCount
                                          baseVertex: 0
                                        baseInstance: actorIdx * MaxVisibleFaces];
            }
        }
    }

    [renderEncoder popDebugGroup];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    // We set the overall vertical field of view to 65 degrees (converted to radians).
    // We store it divided by two.
    static const float fovY_Half = radians_from_degrees(65.0 *.5);
    const float        aspect    = size.width / (float)size.height;

    _cameraFinal.aspectRatio  = aspect;
    _cameraFinal.fovVert_Half = fovY_Half;
    _cameraFinal.distanceNear = 50.f;
    _cameraFinal.distanceFar  = 10000.f;
    
    [self createGBuffer:view];
}

-(void)drawGBuffer:(nonnull MTKView *)view :(id<MTLCommandBuffer>) cmdBuffer
{
    {
        MTLRenderPassDescriptor* gbufferPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
        gbufferPassDesc.colorAttachments[0].texture    = _gbuffer_normal_refl_mask;
        gbufferPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;

        gbufferPassDesc.colorAttachments[1].texture    = _gbuffer_diffuse;
        gbufferPassDesc.colorAttachments[1].loadAction = MTLLoadActionClear;

        gbufferPassDesc.depthAttachment.clearDepth     = 1.0;
        gbufferPassDesc.depthAttachment.loadAction     = MTLLoadActionClear;
        gbufferPassDesc.depthAttachment.storeAction     = MTLStoreActionStore;

        gbufferPassDesc.depthAttachment.texture        = _gbuffer_depth;

        id<MTLRenderCommandEncoder> renderEncoder =
            [cmdBuffer renderCommandEncoderWithDescriptor:gbufferPassDesc];
        renderEncoder.label = @"GBufferPass";
        

        //[renderEncoder setCullMode:MTLCullModeNone];
        [self drawActors: renderEncoder ];

        
        
        [renderEncoder endEncoding];
    }
    

	if(!_SSRTechinque && _SSREnabled)
    {
        id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
        assert(computeEncoder != nil);
        
        computeEncoder.label = @"Generate Hi-Z";
        
        // first copy depthbuffer
        {
            [computeEncoder setComputePipelineState:_depthCopy];
            [computeEncoder setTexture:_gbuffer_depth atIndex:0];
            [computeEncoder setTexture:_depth_hi_z_levels[0] atIndex:1];
            
            MTLSize gridSize = MTLSizeMake(_gbuffer_depth.width,_gbuffer_depth.height, 1);

            // Calculate a threadgroup size.
            MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];
        }
        
        // generate each mip level
        {
            [computeEncoder setComputePipelineState:_createHiZ];
            for(size_t i = 1;i<_depth_hi_z_levels.size();++i)
            {
                [computeEncoder setTexture:_depth_hi_z_levels[i-1] atIndex:0];
                [computeEncoder setTexture:_depth_hi_z_levels[i] atIndex:1];
                
                MTLSize gridSize = MTLSizeMake(_depth_hi_z_levels[i].width,_depth_hi_z_levels[i].height, 1);

                // Calculate a threadgroup size.
                MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);

                // Encode the compute command.
                [computeEncoder dispatchThreads:gridSize
                          threadsPerThreadgroup:threadgroupSize];
            }
                
        }
        

        [computeEncoder updateFence:_gbuffer_fence];
        [computeEncoder endEncoding];
    }

}
-(void)applyLighting:(nonnull MTKView *)view :(id<MTLCommandBuffer>) cmdBuffer
{
    id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
    assert(computeEncoder != nil);
    
    computeEncoder.label = @"Lighting Pass";
    
    // Directional Light
    {
        [computeEncoder pushDebugGroup:@"Directional Lighting"];
        
        [computeEncoder setComputePipelineState:_pipeline_lighting];
        [computeEncoder setTexture:_gbuffer_normal_refl_mask atIndex:0];
        [computeEncoder setTexture:_gbuffer_diffuse atIndex:1];
        [computeEncoder setTexture:_SSREnabled ? _directionalLighting_output : _final_output atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(view.drawableSize.width,view.drawableSize.height, 1);

        // Calculate a threadgroup size.
        MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);

        // Encode the compute command.
        [computeEncoder dispatchThreads:gridSize
                  threadsPerThreadgroup:threadgroupSize];
        [computeEncoder popDebugGroup];
    }
    
    // Screen Space Reflection
    if(_SSREnabled)
    {
        [computeEncoder pushDebugGroup:@"SSR"];
        [computeEncoder waitForFence:_gbuffer_fence];
        
		if(_SSRTechinque)
		{
			[computeEncoder setComputePipelineState:_pipeline_ssr_linear];
			[computeEncoder setTexture:_gbuffer_normal_refl_mask atIndex:0];
			[computeEncoder setTexture:_gbuffer_depth atIndex:1];
			[computeEncoder setTexture:_directionalLighting_output atIndex:2];
			[computeEncoder setTexture:_final_output atIndex:3];
			[computeEncoder setBuffer:_sceneInfoBuffers[_uniformBufferIndex] offset:0 atIndex:0];

			MTLSize gridSize = MTLSizeMake(view.drawableSize.width,view.drawableSize.height, 1);
			
			// Calculate a threadgroup size.
			MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);

			// Encode the compute command.
			[computeEncoder dispatchThreads:gridSize
					  threadsPerThreadgroup:threadgroupSize];
			[computeEncoder popDebugGroup];
		}
		else
		{
			[computeEncoder setComputePipelineState:_pipeline_ssr_hi_z];
			[computeEncoder setTexture:_gbuffer_normal_refl_mask atIndex:0];
			[computeEncoder setTexture:_depth_hi_z atIndex:1];
			[computeEncoder setTexture:_directionalLighting_output atIndex:2];
			[computeEncoder setTexture:_final_output atIndex:3];
			[computeEncoder setBuffer:_sceneInfoBuffers[_uniformBufferIndex] offset:0 atIndex:0];

			MTLSize gridSize = MTLSizeMake(view.drawableSize.width,view.drawableSize.height, 1);
			
			// Calculate a threadgroup size.
			MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);

			// Encode the compute command.
			[computeEncoder dispatchThreads:gridSize
					  threadsPerThreadgroup:threadgroupSize];
			[computeEncoder popDebugGroup];
		}
    }
    [computeEncoder endEncoding];
}
- (void)drawInMTKView:(nonnull MTKView *)view
{
    //------------------------------------------------------------------------------------
    // Update game state and shader parameters

    // Wait to ensure only MaxBuffersInFlight are getting processed by any stage in the Metal
    //   pipeline (App, Metal, Drivers, GPU, etc)
    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);

    // Update the location(s) to which we'll write to in our dynamically changing Metal buffers for
    //   the current frame (i.e. update our slot in the ring buffer used for the current frame)
    _uniformBufferIndex = (_uniformBufferIndex + 1) % MaxBuffersInFlight;

    [self updateGameState];

    //------------------------------------------------------------------------------------
    // Render
    
    {
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        commandBuffer.label = @"Scene Rendering.";
        
        [self drawGBuffer:view :commandBuffer];
        [self applyLighting:view :commandBuffer];
        
        // Commit commands so that Metal can begin working on non-drawable dependant work without
        // waiting for a drawable to become avaliable
        [commandBuffer commit];

    }
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"Drawable Rendering";
    
    // Add a completed handler which signals _inFlightSemaphore when Metal and the GPU has fully
    //   finished processing the commands encoded this frame.  This indicates when the
    //   dynamic buffers written to this frame will no longer be needed by Metal and the GPU.
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
     {
         dispatch_semaphore_signal(block_sema);
     }];
    
    // Obtain the render pass descriptor as late as possible; after updating any buffer state
    //   and rendering any previous passes.
    // By doing this as late as possible, we don't hold onto the drawable any longer than necessary
    //   which could otherwise reduce our frame rate as our application, the GPU, and display all
    //   contend for these limited drawables.
    MTLRenderPassDescriptor* finalPassDescriptor = view.currentRenderPassDescriptor;

    if(finalPassDescriptor != nil)
    {
        id <MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:finalPassDescriptor];

        [renderEncoder setRenderPipelineState:_pipeline_copy];
        
        [renderEncoder setFragmentTexture:_final_output atIndex:0];

        // Draw a quad which fills the screen
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];

        [renderEncoder endEncoding];
    }

    if(view.currentDrawable)
    {
        // Schedule a present once the framebuffer is complete using the current drawable
        [commandBuffer presentDrawable:view.currentDrawable];
    }

    // Finalize rendering here & push the command buffer to the GPU
    [commandBuffer commit];
}

@end

/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Header for mesh and submesh objects used for managing models
*/

#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

// App specific submesh class containing data to draw a submesh
@interface AAPLSubmesh : NSObject

// A MetalKit submesh mesh containing the primitive type, index buffer, and index count
//   used to draw all or part of its parent AAPLMesh object
@property (nonatomic, readonly, nonnull) MTKSubmesh *metalKitSubmmesh;

// Material textures (indexed by AAPLTextureIndex) to set in the Metal render command encoder
//  before drawing the submesh
@property (nonatomic, readonly, nonnull) NSArray<id<MTLTexture>> *textures;

@end

// App specific mesh class containing vertex data describing the mesh and submesh object describing
//   how to draw parts of the mesh
@interface AAPLMesh : NSObject

+ (nullable NSArray<AAPLMesh*> *)newMeshesFromObject:(nonnull MDLObject*)object
                             modelIOVertexDescriptor:(nonnull MDLVertexDescriptor*)vertexDescriptor
                               metalKitTextureLoader:(MTKTextureLoader*_Nullable)textureLoader
                                         metalDevice:(nonnull id<MTLDevice>)device
                                               error:(NSError * __nullable * __nullable)error;

// Constructs an array of meshes from the provided file URL, which indicate the location of a model
//  file in a format supported by Model I/O, such as OBJ, ABC, or USD.  mdlVertexDescriptor defines
//  the layout Model I/O will use to arrange the vertex data while the bufferAllocator supplies
//  allocations of Metal buffers to store vertex and index data
+ (nullable NSArray<AAPLMesh *> *)newMeshesFromUrl:(nonnull NSURL *)url
                           modelIOVertexDescriptor:(nonnull MDLVertexDescriptor *)vertexDescriptor
                                       metalDevice:(nonnull id<MTLDevice>)device
                                             error:(NSError * __nullable * __nullable)error
                                              aabb:(MDLAxisAlignedBoundingBox&)aabb;

// A MetalKit mesh containing vertex buffers describing the shape of the mesh
@property (nonatomic, readonly, nonnull) MTKMesh *metalKitMesh;

// An array of AAPLSubmesh objects containing buffers and data with which we can make a draw call
//  and material data to set in a Metal render command encoder for that draw call
@property (nonatomic, readonly, nonnull) NSArray<AAPLSubmesh*> *submeshes;

@end

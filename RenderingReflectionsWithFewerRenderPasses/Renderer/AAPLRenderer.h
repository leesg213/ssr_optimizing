/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Header for renderer class which performs Metal setup and per frame rendering
*/

#import <MetalKit/MetalKit.h>

@interface AAPLRenderer : NSObject<MTKViewDelegate>

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView;

-(void)onToggleSSRButton:(NSButton *)button;
-(void)onToggleSSRTechniqueButton:(NSButton *)button;
-(void)onMoveCamPosSlider:(NSSlider *)slider;
-(void)onAnimSpeedSlider:(NSSlider *)slider;
@end

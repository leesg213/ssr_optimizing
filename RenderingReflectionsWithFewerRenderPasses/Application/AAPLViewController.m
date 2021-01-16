/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of the cross-platform view controller
*/

#import "AAPLViewController.h"
#import "AAPLRenderer.h"

@implementation AAPLViewController
{
    MTKView *_view;

    AAPLRenderer *_renderer;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    // Set the view to use the default device
    _view = (MTKView *)self.view;
    
    _view.device = MTLCreateSystemDefaultDevice();
    
    NSAssert(_view.device, @"Metal is not supported on this device");

#if TARGET_IOS
    BOOL supportsLayerSelection = NO;

    supportsLayerSelection = [_view.device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily5_v1];

    NSAssert(supportsLayerSelection, @"Sample requires iOS_GPUFamily5_v1 for Layer Selection");
#endif
    
    _renderer = [[AAPLRenderer alloc] initWithMetalKitView:_view];
    
    NSAssert(_renderer, @"Renderer failed initialization");
    
    [_renderer mtkView:_view drawableSizeWillChange:_view.drawableSize];

    _view.delegate = _renderer;
}

#if defined(TARGET_IOS)
- (BOOL)prefersHomeIndicatorAutoHidden
{
    return YES;
}
#endif

- (IBAction)On_SSR_ToggleButton:(NSButton *)sender {
	[_renderer onToggleSSRButton:sender];
}
- (IBAction)On_SSR_ToggleTechniqueButton:(NSButton *)sender {
	[_renderer onToggleSSRTechniqueButton:sender];
}
- (IBAction)OnMoveCamPosScrollBar:(NSSlider *)sender {
	[_renderer onMoveCamPosSlider:sender];
}
- (IBAction)OnAnimSpeedSlider:(NSSlider *)sender {
	[_renderer onAnimSpeedSlider:sender];
}

@end

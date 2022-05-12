#ifndef DETECTOR_POST_PROCESSING_HPP
#define DETECTOR_POST_PROCESSING_HPP

#include "UseCaseCommonUtils.hpp"
#include "DetectionResult.hpp"

#define DISPLAY_RGB_IMAGE (1)

#define INPUT_IMAGE_WIDTH 160
#define INPUT_IMAGE_HEIGHT 160

#define ORIGINAL_IMAGE_WIDTH 160
#define ORIGINAL_IMAGE_HEIGHT 160

#if DISPLAY_RGB_IMAGE
#define FORMAT_MULTIPLY_FACTOR 3
#else
#define FORMAT_MULTIPLY_FACTOR 1
#endif /* DISPLAY_RGB_IMAGE */

void runPostProcessing(uint8_t *img_in,TfLiteTensor* model_output[2],std::vector<arm::app::DetectionResult> & results_out);

void draw_box_on_image(uint8_t *img_in,int im_w,int im_h,int bx,int by,int bw,int bh);
void rgb_to_grayscale(const uint8_t *rgb,uint8_t *gray, int im_w,int im_h);



#define SSD_NUM_PRIORS 1118
#define SSD_NUM_FEATURE_MAPS 4
#define SSD_NUM_MAX_TARGETS 20
#define WITH_YAW 0
#define WITH_YAW_AND_LANDMARKS 1



namespace arm {
namespace app {

namespace ssd {

typedef struct EmzaColor {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} EmzaColor;

typedef struct FaceInfo {
    short x1;
    short y1;
    short x2;
    short y2;
    #if WITH_YAW
    short yaw;
    #endif // WITH_YAW
    #if WITH_YAW_AND_LANDMARKS
    short yaw;
    short landmarks[10]; // (x,y ... ,x,y)
    #endif // WITH_YAW_AND_LANDMARKS
    float score;
} FaceInfo;


short generateBBox(TfLiteTensor* model_output[], FaceInfo out_boxes[]);

short nms(FaceInfo *input_boxes, short num_input_boxes, FaceInfo output_boxes[]);

void drawLandmarksAndYaw(uint8_t* srcPtr,int srcHeight, int srcWidth,
                         FaceInfo *face_list, int n_faces,int n_channels,float scale_factor);

} /* namepsace ssd */

} /* namespace app */
} /* namespace arm */

#endif /* DETECTOR_POST_PROCESSING_HPP */

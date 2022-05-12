/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "UseCaseHandler.hpp"
#include "InputFiles.hpp"
#include "YoloFastestModel.hpp"
#include "UseCaseCommonUtils.hpp"
#include "DetectionUseCaseUtils.hpp"
#include "DetectorPostProcessing.h"
#include "hal.h"

#include <inttypes.h>


#define SHOW_TEXT (0)

//TODO: make path relative before uploading to github
extern "C" {
#include "/home/smishash/Alif/ensembleML-alif/source/application/hal/platforms/ensemble/viewfinder/include/image_processing.h"
#include "lvgl.h"

extern int wb_a;
extern int wb_b;
extern int wb_c;
}

extern ARM_DRIVER_GPIO Driver_GPIO1;
/* used for presentation, original images are read-only"*/

//static uint8_t g_image_buffer[DISPLAY_W*DISPLAY_H*RGB_BYTES] __attribute__((aligned (16)));
extern uint8_t g_image_buffer[DISPLAY_W*DISPLAY_H*RGB_BYTES];
extern uint8_t rgb_image[CIMAGE_X*CIMAGE_Y*RGB_BYTES];

extern lv_obj_t *labelResult1;

namespace arm {
namespace app {


hal_platform platform_g ;
    /* Object detection classification handler. */
    bool ObjectDetectionHandler(ApplicationContext& ctx, uint32_t imgIndex, bool runAll)
    {
    	hal_platform& platform = ctx.Get<hal_platform&>("platform");
    	platform_g = platform;
        auto& profiler = ctx.Get<Profiler&>("profiler");

        constexpr uint32_t dataPsnImgDownscaleFactor = 1;
        constexpr uint32_t dataPsnImgStartX = 10;
        constexpr uint32_t dataPsnImgStartY = 35;

        constexpr uint32_t dataPsnTxtInfStartX = 150;
        constexpr uint32_t dataPsnTxtInfStartY = 40;

        platform.data_psn->clear(COLOR_BLACK);

        auto& model = ctx.Get<Model&>("model");

        /* If the request has a valid size, set the image index. */
        if (imgIndex < NUMBER_OF_FILES) {
            if (!SetAppCtxIfmIdx(ctx, imgIndex, "imgIndex")) {
                return false;
            }
        }
        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        auto curImIdx = ctx.Get<uint32_t>("imgIndex");

        TfLiteTensor* inputTensor = model.GetInputTensor(0);

        if (!inputTensor->dims) {
            printf_err("Invalid input tensor dims\n");
            return false;
        } else if (inputTensor->dims->size < 3) {
            printf_err("Input tensor dimension should be >= 3\n");
            return false;
        }

        TfLiteIntArray* inputShape = model.GetInputShape(0);

        const uint32_t nCols = inputShape->data[arm::app::YoloFastestModel::ms_inputColsIdx];
        const uint32_t nRows = inputShape->data[arm::app::YoloFastestModel::ms_inputRowsIdx];
        const uint32_t nPresentationChannels = FORMAT_MULTIPLY_FACTOR;
               
        std::vector<DetectionResult> results;

        float scale_factor = float(DISPLAY_W)/float(MIMAGE_X);
        char str_out[30];

        do {
            /* Strings for presentation/logging. */
            std::string str_inf{"Running inference... "};

            
            //From memory, get buffer
            const uint8_t* curr_image = get_img_array(ctx.Get<uint32_t>("imgIndex"));

#if FROM_CAMERA
            //From camera, override
            //const uint8_t* curr_image=0;
            platform.data_acq->get_data((int)(curr_image));
#else
            memcpy(g_image_buffer,curr_image, MIMAGE_X*MIMAGE_Y*FORMAT_MULTIPLY_FACTOR); //good
#endif


            rgb_to_grayscale(curr_image,inputTensor->data.uint8,nCols,nRows);
                       

            /* Display this image on the LCD. */            
//            platform.data_psn->present_data_image(
//            		g_image_buffer,
//					DISPLAY_W,DISPLAY_H,nPresentationChannels,
//                dataPsnImgStartX, dataPsnImgStartY, dataPsnImgDownscaleFactor);

            /* If the data is signed. */
            if (model.IsDataSigned()) {
                image::ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
            }

            /* Display message on the LCD - inference running. */
//            platform.data_psn->present_data_text(str_inf.c_str(), str_inf.size(),
//                                    dataPsnTxtInfStartX, dataPsnTxtInfStartY, 0);

            /* Run inference over this image. */
//            info("Running inference on image %" PRIu32 " => %s\n", ctx.Get<uint32_t>("imgIndex"),
//                get_filename(ctx.Get<uint32_t>("imgIndex")));

            //Set profiling pin high
            Driver_GPIO1.SetValue(PIN_NUMBER_15, GPIO_PIN_OUTPUT_STATE_HIGH);// PROFILING_PIN_NUMBER
            platform.timer->reset();

            if (!RunInference(model, profiler)) {
                return false;
            }

            //Set profiling pin low - profiling inference only
            Driver_GPIO1.SetValue(PIN_NUMBER_15, GPIO_PIN_OUTPUT_STATE_LOW); //PROFILING_PIN_NUMBER

            /* Erase. */
//            str_inf = std::string(str_inf.size(), ' ');
//            platform.data_psn->present_data_text(str_inf.c_str(), str_inf.size(),
//                                    dataPsnTxtInfStartX, dataPsnTxtInfStartY, 0);
                                    
            /* Dtector post-processng*/
//            TfLiteTensor* output_arr[2] = {nullptr,nullptr};
//            output_arr[0] = model.GetOutputTensor(0);
//            output_arr[1] = model.GetOutputTensor(1);
//            runPostProcessing(g_image_buffer,output_arr,results);


            /* SSD post-processing */

#if WITH_YAW_AND_LANDMARKS
            TfLiteTensor* modelOutput[4] = { nullptr,nullptr, nullptr, nullptr };
            modelOutput[0] = model.GetOutputTensor(0); // Landmarks
            modelOutput[1] = model.GetOutputTensor(1); // bbox
            modelOutput[2] = model.GetOutputTensor(2); // yaw
            modelOutput[3] = model.GetOutputTensor(3); // class
#elif WITH_YAW
            TfLiteTensor* modelOutput[3] = { nullptr,nullptr, nullptr };
            modelOutput[0] = model.GetOutputTensor(0);
            modelOutput[1] = model.GetOutputTensor(2);
            modelOutput[2] = model.GetOutputTensor(1);
#else
            TfLiteTensor* modelOutput[2] = { nullptr,nullptr };
            modelOutput[0] = model.GetOutputTensor(0);
            modelOutput[1] = model.GetOutputTensor(1);
#endif

            short num_detections = 0;
            ssd::FaceInfo bboxes[SSD_NUM_MAX_TARGETS] = {};
            ssd::FaceInfo face_list[SSD_NUM_MAX_TARGETS] = {};
            num_detections = ssd::generateBBox(modelOutput, bboxes);
            
            //info("generateBBox, num_detections = %d \n",num_detections);

            if (num_detections > 1)
                num_detections = ssd::nms(bboxes, num_detections, face_list);
            else if (num_detections == 1)
                face_list[0] = bboxes[0];

            //num_detections valid now

            //Set profiling pin high - profiling includes post-processing
            //Driver_GPIO1.SetValue(PIN_NUMBER_15, GPIO_PIN_OUTPUT_STATE_HIGH); //PROFILING_PIN_NUMBER
            


            info("nms done, num_detections = %d \n",num_detections);
            
            ssd::drawLandmarksAndYaw(g_image_buffer,DISPLAY_H, DISPLAY_W,
                                    face_list, num_detections,nPresentationChannels,scale_factor);


            
            
            platform.data_psn->present_data_image(
            		g_image_buffer,
					DISPLAY_W,DISPLAY_H, nPresentationChannels,
                dataPsnImgStartX, dataPsnImgStartY, dataPsnImgDownscaleFactor);
            
            /*Detector post-processng*/

#if SHOW_TEXT
            //Show text in percents
            for (int i=0; i < num_detections; i++) {

            	//clear
                memset(str_out,0x20,sizeof(str_out)); //space
                str_out[sizeof(str_out)-1] =0; //null
                str_out[sizeof(str_out)-2] =0; //null
                lv_label_set_text_fmt(labelResult1, "%s",str_out);

                //draw

                //sprintf(str_out,"%d) Yaw = %d\%",i+1,face_list[i].yaw);
                sprintf(str_out,"WB: %d,%d,%d;",wb_a,wb_b,wb_c);
                platform.data_psn->present_data_text(str_out,strlen(str_out),0, 0 + i*20 , 0);


                lv_label_set_text_fmt(labelResult1, "%s",str_out);

                memset(str_out,0,sizeof(str_out));
                sprintf(str_out,"%d",i+1);
                platform.data_psn->present_data_text(str_out,strlen(str_out),dataPsnImgStartX + face_list[i].x1,dataPsnImgStartY + face_list[i].y1, 0);
            }
#endif



            /* Add results to context for access outside handler. */
            ctx.Set<std::vector<DetectionResult>>("results", results);

#if VERIFY_TEST_OUTPUT
            arm::app::DumpTensor(outputTensor);
#endif /* VERIFY_TEST_OUTPUT */

            if (!image::PresentInferenceResult(platform, results)) {
                return false;
            }

            profiler.PrintProfilingResult();

            //IncrementAppCtxIfmIdx(ctx,"imgIndex");

            //Set profiling pin low - back to initial state
            //Driver_GPIO1.SetValue(PIN_NUMBER_15, GPIO_PIN_OUTPUT_STATE_LOW); //PROFILING_PIN_NUMBER

        } while (runAll && ctx.Get<uint32_t>("imgIndex") != curImIdx);

        return true;
    }
 


} /* namespace app */
} /* namespace arm */

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
#include "hal.h"                    /* Brings in platform definitions. */
#include "InputFiles.hpp"           /* For input images. */
#include "YoloFastestModel.hpp"     /* Model class for running inference. */
#include "UseCaseHandler.hpp"       /* Handlers for different user options. */
#include "UseCaseCommonUtils.hpp"   /* Utils functions. */
#include "DetectionUseCaseUtils.hpp"   /* Utils functions specific to object detection. */

extern ARM_DRIVER_GPIO Driver_GPIO1;

void main_loop(hal_platform& platform)
{
    arm::app::YoloFastestModel model;  /* Model wrapper object. */

    /* Load the model. */
    if (!model.Init()) {
        printf_err("Failed to initialise model\n");
        return;
    }

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{&platform, "object_detection"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<hal_platform&>("platform", platform);
    caseContext.Set<arm::app::Model&>("model", model);
    caseContext.Set<uint32_t>("imgIndex", 0);

    /*Profioing pin setup 	*/
    Driver_GPIO1.Initialize(PROFILING_PIN_NUMBER,NULL);
	Driver_GPIO1.PowerControl(PROFILING_PIN_NUMBER,  ARM_POWER_FULL);
	Driver_GPIO1.SetDirection(PROFILING_PIN_NUMBER, GPIO_PIN_DIRECTION_OUTPUT);
	PINMUX_Config (PORT_NUMBER_1, PROFILING_PIN_NUMBER, PINMUX_ALTERNATE_FUNCTION_0);
	PINPAD_Config (PORT_NUMBER_1, PROFILING_PIN_NUMBER, (0x09 | PAD_FUNCTION_OUTPUT_DRIVE_STRENGTH_04_MILI_AMPS));

	Driver_GPIO1.SetValue(PROFILING_PIN_NUMBER, GPIO_PIN_OUTPUT_STATE_LOW);
    
    /* Loop. */
    bool executionSuccessful = true;
    constexpr bool bUseMenu = NUMBER_OF_FILES > 1 ? true : false;

    /* Loop. */
    do {
    	//Set profiling pin low
    	Driver_GPIO1.SetValue(PROFILING_PIN_NUMBER, GPIO_PIN_OUTPUT_STATE_LOW);

        int menuOption = common::MENU_OPT_RUN_INF_NEXT;
        if (bUseMenu) {
            DisplayDetectionMenu();
            menuOption = arm::app::ReadUserInputAsInt(platform);
            printf("\n");
        }
			menuOption = common::MENU_OPT_RUN_INF_NEXT;			
        switch (menuOption) {
            case common::MENU_OPT_RUN_INF_NEXT:
                executionSuccessful = ObjectDetectionHandler(caseContext, caseContext.Get<uint32_t>("imgIndex"), false);
                break;
            case common::MENU_OPT_RUN_INF_CHOSEN: {
                printf("    Enter the image index [0, %d]: ", NUMBER_OF_FILES-1);
                fflush(stdout);
                auto imgIndex = static_cast<uint32_t>(arm::app::ReadUserInputAsInt(platform));
                executionSuccessful = ObjectDetectionHandler(caseContext, imgIndex, false);
                break;
            }
            case common::MENU_OPT_RUN_INF_ALL:
                executionSuccessful = ObjectDetectionHandler(caseContext, caseContext.Get<uint32_t>("imgIndex"), true);
                break;
            case common::MENU_OPT_SHOW_MODEL_INFO:
                executionSuccessful = model.ShowModelInfoHandler();
                break;
            case common::MENU_OPT_LIST_IFM:
                executionSuccessful = ListFilesHandler(caseContext);
                break;
            default:
                printf("Incorrect choice, try again.");
                break;
        }
    } while (executionSuccessful && bUseMenu);
    info("Main loop terminated.\n");
}

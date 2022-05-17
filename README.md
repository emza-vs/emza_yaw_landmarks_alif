## emza_yaw_landmarks_alif
## ARM armclang compiler is required to run the example.
## To run the example, get the ML repo ALIF, and put it into ~/demo/ensembleML-alif folder

## clone this repo:
cd ~/demo

git clone https://github.com/emza-vs/emza_yaw_landmarks_alif.git

## Merge the ml-embedded-evaluation-kit folder from  emza_yaw_landmarks_alif into the ensembleML-alif folder from Alif,overwrite the files.

## Now build the example

cd ~/demo

mkdir build

cd build

cmake ../ensembleML-alif -DTARGET_PLATFORM=ensemble -DTARGET_SUBSYSTEM=RTSS-HP -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-armclang.cmake -DUSE_CASE_BUILD=object_detection  -Dobject_detection_IMAGE_SIZE=160 -Dobject_detection_MODEL_TFLITE_PATH=resources_downloaded/object_detection/ssd_slim_120x160x1_yaw_landmarks_v3_int8_vela_H256.tflite -DCMAKE_BUILD_TYPE=Release -DLOG_LEVEL=LOG_LEVEL_ERROR  -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.15


make

## Flash the ./bin/ethos-u-object_detection.axf file onto the Alif development board following the flashing documentation.

## NOTE: for detailed step-by step instruction how to set-up the Python toolchanin please look at the readme file here: 

https://github.com/emza-vs/face_detection_example_arm_u55/blob/master/README.md
 


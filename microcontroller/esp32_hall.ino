#include <Wire.h>
#include <AS5600.h>
#include <Dynamixel2Arduino.h>
#include <Preferences.h>

/*---------------------- ESP32 defines and variables -------------------*/
#define I2C_SDA 18
#define I2C_SCL 19

/*---------------------- DXL defines and variables ---------------------*/

#define DXL_DIR_PIN 22
#define DXL_PROTOCOL_VER_2_0 2.0
#define DXL_MODEL_NUM 0xbaff
#define DEFAULT_ID 42
#define DEFAULT_BAUD 6 //4mbaud

uart_t* uart;
DYNAMIXEL::FastSlave dxl(DXL_MODEL_NUM, DXL_PROTOCOL_VER_2_0);

#define ADDR_CONTROL_ITEM_BAUD 8

#define ADDR_CONTROL_ITEM_MAGNITUDE 10
#define ADDR_CONTROL_ITEM_ANGLE 11

#define ADDR_CONTROL_ITEM_MAGNET_STR 20
#define ADDR_CONTROL_ITEM_AGC 21


//1x: sensor output, 2x: debug output, 3x: inputs
/*---------------------- DXL ---------------------*/

// id and baud are stored using preferences for persistence between resets of the chip
uint8_t id;
uint8_t baud;

uint32_t dxl_to_real_baud(uint8_t baud)
{
    int real_baud = 57600;
    switch(baud)
    {
        case 0: real_baud = 9600; break;
        case 1: real_baud = 57600; break;
        case 2: real_baud = 115200; break;
        case 3: real_baud = 1000000; break;
        case 4: real_baud = 2000000; break;
        case 5: real_baud = 3000000; break;
        case 6: real_baud = 4000000; break;
        case 7: real_baud = 4500000; break;
    }
    return real_baud;
}

/*---------------------- Setup ---------------------*/

// define two tasks for reading the dxl bus and doing other work
void TaskDXL( void *pvParameters );
void TaskWorker( void *pvParameters );

TaskHandle_t th_dxl,th_worker;

Preferences dxl_prefs;
Preferences hall_prefs;

// define sensor
AMS_5600 ams5600;

void setup() {
    disableCore0WDT(); // required since we dont want FreeRTOS to slow down our reading if the Wachdogtimer (WTD) fires
    disableCore1WDT();
    xTaskCreatePinnedToCore(
            TaskDXL
            ,  "TaskDXL"   // A name just for humans
            ,  65536  // This stack size can be checked & adjusted by reading the Stack Highwater
            ,  NULL
            ,  3  // Priority 3 since otherwise
            ,  &th_dxl
            ,  0);

    xTaskCreatePinnedToCore(
            TaskWorker
            ,  "TaskWork"
            ,  65536  // Stack size
            ,  NULL
            ,  3  // Priority
            ,  &th_worker
            ,  1);
}

void loop() {
    //tasks only
}

/*---------------------- Tasks ---------------------*/
void TaskDXL(void *pvParameters) {

}

void TaskWorker(void *pvParameters) {

}

/*---------------------- Hall Sensor ---------------------*/
float rawToDeg(word rawAngle) {
    // sensor returns 0-4095, so 1 equals 0.087 degrees.
    return rawAngle * 0.087;
}
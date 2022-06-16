#include <Wire.h>
#include <Dynamixel2Arduino.h>
#include <Preferences.h>
#include "fast_slave.h"


/*---------------------- DXL defines and variables ---------------------*/

#define DXL_DIR_PIN 22
#define DXL_PROTOCOL_VER_2_0 2.0
#define DXL_MODEL_NUM 0xabcd
#define DEFAULT_ID 82
#define DEFAULT_BAUD 6 //2mbaud. TODO: Might need to reduce to 4 = 2mbaud, because the flashing tool cant handle more.

#define I2C_SDA 18
#define I2C_SCL 19

uart_t* uart;
DYNAMIXEL::FastSlave dxl(DXL_MODEL_NUM, DXL_PROTOCOL_VER_2_0);

#define ADDR_CONTROL_ITEM_BAUD 8

#define ADDR_CONTROL_ITEM_ANGLE_RAW 34 //4

//3x: sensor output, 5x: debug output, 6x: inputs

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

/*--------------- Hall Sensor Variables ------------*/
uint32_t angleRaw;

/*---------------------- Setup ---------------------*/

// define two tasks for reading the dxl bus and doing other work
void TaskDXL( void *pvParameters );
void TaskWorker( void *pvParameters );

TaskHandle_t th_dxl,th_worker;

Preferences dxl_prefs;

void setup() {
    disableCore0WDT(); // required since we dont want FreeRTOS to slow down our reading if the Wachdogtimer (WTD) fires
    disableCore1WDT();
    //Serial.begin(2000000); debug only
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
    (void) pvParameters;

    dxl_prefs.begin("dxl");
    if(!dxl_prefs.getUChar("init")) // check if prefs are initialized
    {
        dxl_prefs.putUChar("id", DEFAULT_ID);
        dxl_prefs.putUChar("baud", DEFAULT_BAUD);
        dxl_prefs.putUChar("init",1); // set initialized
    }
    id = dxl_prefs.getUChar("id");
    baud = dxl_prefs.getUChar("baud");

    dxl.setPortProtocolVersion(DXL_PROTOCOL_VER_2_0);
    dxl.setFirmwareVersion(1);
    dxl.setID(id);

    dxl.addControlItem(ADDR_CONTROL_ITEM_BAUD, baud);

    dxl.addControlItem(ADDR_CONTROL_ITEM_ANGLE_RAW, angleRaw);

    dxl.setWriteCallbackFunc(write_callback_func);

    pinMode(DXL_DIR_PIN, OUTPUT);
    // init uart 0, given baud, 8bits 1stop no parity, pin 3 and 1, 256 buffer, no inversion
    uart = uartBegin(0, dxl_to_real_baud(baud), SERIAL_8N1, 3, 1,  256, false);
    // disable all interrupts
    uart->dev->conf1.rx_tout_en = 0;
    uart->dev->int_ena.rxfifo_full = 0;
    uart->dev->int_ena.frm_err = 0;
    uart->dev->int_ena.rxfifo_tout = 0;

    for (;;)
    {
        if(dxl.processPacket(uart)){
            if(dxl.getID() != id) // since we cant add the id as a control item, we need to check if it has been updated manually
            {
                id = dxl.getID();
                dxl_prefs.putUChar("id", id);
            }
        }
    }
}

void write_callback_func(uint16_t item_addr, uint8_t &dxl_err_code, void* arg) {
    (void)dxl_err_code, (void)arg;
    if (item_addr == ADDR_CONTROL_ITEM_BAUD)
    {
        dxl_prefs.putUChar("baud", baud);
        ESP.restart(); // restart whole chip since restarting serial port crashes esp
    }
}

void TaskWorker(void *pvParameters) {
  pinMode(17, INPUT);
  //Wire.begin(I2C_SDA, I2C_SCL);
  //Wire.beginTransmission(0x36);
  //Wire.write(0x08);
  //Wire.write(0b10100000); //set: 460 Hz, PWM mode, no hysteresis, normal power mode
  //Wire.endTransmission();
    for (;;) {
        angleRaw = pulseIn(17, HIGH); //convert angle value from unsigned long to uint32
        //Serial.println(angleRaw); Debug only
    }
}

# deepSEA
Software for correcting positional and dynamic errors introduced by Series Elastic Actuators using neural networks. Developed in the context of humanoid robots. Part of my master thesis. 

## Requirements

This software requires parts of the Hamburg Bit-Bots codebase to run. It can be found at https://github.com/bit-bots/bitbots_meta. 

## Circuit Board

The circuit board files for the Hall sensor, as well as the BOM can be found in the folder 'electonics'. Code for the microcontroller can be found in the folder 'microcontroller'.

## CAD Models

CAD models can be accessed at 
* https://cad.onshape.com/documents/520e275ca0a4ee1da79ab213/w/c9ea3b5bef5fc27494e809fb/e/f1a5e02c80f92696ed8da6b5?renderMode=0&uiState=63512a97a10e15610f2da196
* https://cad.onshape.com/documents/725c43c79f6ce8c6c15772b9/w/ed6020920787d50c3c2799e9/e/e051bd42fbad81ebb299f2ea?renderMode=0&uiState=63512aad57c0844ed4e72b1e

## Acknowledgements

Contents of the 'bitbots_lowlevel' and 'bitbots_msgs' folders originally stem from the Bit-Bots codebase, but have been modified to fit this usecase.
Mainly, an interface and a converter for the Hall sensor, as well as a new message type were added. The original code can be found here:

* https://github.com/bit-bots/bitbots_lowlevel
* https://github.com/bit-bots/bitbots_msgs

The folder 'microcontroller' contains the 'fast_slave' class. This was also originally developed by the Hamburg Bit-Bots, based on the Dynamixel2Arduino library. The original files can be found here:

* https://github.com/bit-bots/bit_foot


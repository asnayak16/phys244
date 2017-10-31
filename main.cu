//*****************************************************************************
//PHYS 244							main.cu	       				14th June 2017
//-----------------------------------------------------------------------------
//Author: Ganesh Ranganath Chandrasekar Iyer, Ashwin Nayak, Prashant Singh
//Objective: Implements the Marching Cube Algorithm using Nvidia's GP-GPU 
//Funtions: StatusLog, input_parameter_initialization, inputRead, 
//			marchingCubes,kernel,linearInterpolation, GetMemoryStatus
//          createOutputList,writeOutputList
//Compiler: nvcc
//Note: x64 Relese, compute_30,sm_30, /STACK:2000000, /HEAP:2000000
//Input: Name of the file which has the input parameters in command line
//		 iso_data.dat: File which has the volumetric data
//*****************************************************************************

/******************************************************************************
Disclaimer: The Tables are based on Cory Gene Bloyd formulation and algorithm
is based on Paul Bourke's article.

For additional details refer: http://paulbourke.net/geometry/polygonise/

Reference: Lorenson W.E. and Cline H.E., ‘Marching Cubes: A High-Resolution 3D 
		   Surface Construction Algorithm’, Computer Graphics (SIGGRAPH Proceed
		   ings), 1987, Vol 21-4, 163-169.
		   Suh, J.W., Kim, Y., ‘Accelerating MATLAB with GPU Computing’, 2013.
******************************************************************************/

//-----------------------------------------------------------------------------
// Including Header Files

#include"cuda_runtime.h"
#include"device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <string>
#include <numeric>

#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include<thrust/sequence.h>
#include<thrust/execution_policy.h>
#include<thrust/transform.h>
#include"vector_types.h"
#include"vector_functions.h"
#include<thrust/device_ptr.h>
#include<thrust/system_error.h>

//-----------------------------------------------------------------------------
//Initialisation of Constants and Global Variables
//-----------------------------------------------------------------------------

//Maximum vertices 15
int MAX_VERTICES = 15;
//Initialising for use in GPU
__constant__ int MAX_VERTEX = 15;

/*
Output bin - contains unformatted vertex list
       counter - contains number of traingles for each voxel		
*/
float3* bin;
int* counter;

__constant__ unsigned int edgeTable[256] =
{
	0,  265,  515,  778, 1030, 1295, 1541, 1804,
	2060, 2309, 2575, 2822, 3082, 3331, 3593, 3840,
	400,  153,  915,  666, 1430, 1183, 1941, 1692,
	2460, 2197, 2975, 2710, 3482, 3219, 3993, 3728,
	560,  825,   51,  314, 1590, 1855, 1077, 1340,
	2620, 2869, 2111, 2358, 3642, 3891, 3129, 3376,
	928,  681,  419,  170, 1958, 1711, 1445, 1196,
	2988, 2725, 2479, 2214, 4010, 3747, 3497, 3232,
	1120, 1385, 1635, 1898,  102,  367,  613,  876,
	3180, 3429, 3695, 3942, 2154, 2403, 2665, 2912,
	1520, 1273, 2035, 1786,  502,  255, 1013,  764,
	3580, 3317, 4095, 3830, 2554, 2291, 3065, 2800,
	1616, 1881, 1107, 1370,  598,  863,   85,  348,
	3676, 3925, 3167, 3414, 2650, 2899, 2137, 2384,
	1984, 1737, 1475, 1226,  966,  719,  453,  204,
	4044, 3781, 3535, 3270, 3018, 2755, 2505, 2240,
	2240, 2505, 2755, 3018, 3270, 3535, 3781, 4044,
	204,  453,  719,  966, 1226, 1475, 1737, 1984,
	2384, 2137, 2899, 2650, 3414, 3167, 3925, 3676,
	348,   85,  863,  598, 1370, 1107, 1881, 1616,
	2800, 3065, 2291, 2554, 3830, 4095, 3317, 3580,
	764, 1013,  255,  502, 1786, 2035, 1273, 1520,
	2912, 2665, 2403, 2154, 3942, 3695, 3429, 3180,
	876,  613,  367,  102, 1898, 1635, 1385, 1120,
	3232, 3497, 3747, 4010, 2214, 2479, 2725, 2988,
	1196, 1445, 1711, 1958,  170,  419,  681,  928,
	3376, 3129, 3891, 3642, 2358, 2111, 2869, 2620,
	1340, 1077, 1855, 1590,  314,   51,  825,  560,
	3728, 3993, 3219, 3482, 2710, 2975, 2197, 2460,
	1692, 1941, 1183, 1430,  666,  915,  153,  400,
	3840, 3593, 3331, 3082, 2822, 2575, 2309, 2060,
	1804, 1541, 1295, 1030,  778,  515,  265,    0
};

__constant__ int triTable[256][16] =
{
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1 },
	{ 8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1 },
	{ 3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1 },
	{ 4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1 },
	{ 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1 },
	{ 9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1 },
	{ 10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1 },
	{ 5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1 },
	{ 5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1 },
	{ 8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1 },
	{ 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1 },
	{ 7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1 },
	{ 2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1 },
	{ 11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1 },
	{ 5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1 },
	{ 11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1 },
	{ 11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1 },
	{ 2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1 },
	{ 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1 },
	{ 3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1 },
	{ 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1 },
	{ 6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1 },
	{ 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1 },
	{ 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1 },
	{ 3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1 },
	{ 0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1 },
	{ 9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1 },
	{ 8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1 },
	{ 5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1 },
	{ 0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1 },
	{ 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1 },
	{ 10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1 },
	{ 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1 },
	{ 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1 },
	{ 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1 },
	{ 3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1 },
	{ 6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1 },
	{ 9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1 },
	{ 8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1 },
	{ 3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1 },
	{ 6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1 },
	{ 10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1 },
	{ 10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1 },
	{ 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1 },
	{ 7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1 },
	{ 7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1 },
	{ 2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1 },
	{ 1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1 },
	{ 11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1 },
	{ 8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1 },
	{ 0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1 },
	{ 7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1 },
	{ 7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1 },
	{ 2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1 },
	{ 10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1 },
	{ 0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1 },
	{ 7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1 },
	{ 6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1 },
	{ 6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1 },
	{ 4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1 },
	{ 10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1 },
	{ 8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1 },
	{ 1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1 },
	{ 10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1 },
	{ 10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1 },
	{ 9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1 },
	{ 7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1 },
	{ 3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1 },
	{ 7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1 },
	{ 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1 },
	{ 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1 },
	{ 9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1 },
	{ 1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1 },
	{ 4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1 },
	{ 7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1 },
	{ 6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1 },
	{ 0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1 },
	{ 6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1 },
	{ 0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1 },
	{ 11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1 },
	{ 6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1 },
	{ 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1 },
	{ 9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1 },
	{ 1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1 },
	{ 10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1 },
	{ 0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1 },
	{ 10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1 },
	{ 11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1 },
	{ 9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1 },
	{ 7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1 },
	{ 2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1 },
	{ 8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1 },
	{ 9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1 },
	{ 9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1 },
	{ 1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1 },
	{ 9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1 },
	{ 0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1 },
	{ 10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1 },
	{ 2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1 },
	{ 0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5 , 1, 11, -1 },
	{ 0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1 },
	{ 9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1 },
	{ 5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1 },
	{ 5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1 },
	{ 8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1 },
	{ 9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1 },
	{ 1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1 },
	{ 3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1 },
	{ 4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1 },
	{ 9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1 },
	{ 11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1 },
	{ 11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1 },
	{ 2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1 },
	{ 9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1 },
	{ 3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1 },
	{ 1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1 },
	{ 4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1 },
	{ 0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1 },
	{ 1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
};
//-----------------------------------------------------------------------------
/*****************************************************************************/
//-----------------------------------------------------------------------------
//Declaring Functions and Structures
//-----------------------------------------------------------------------------
/*
* Name: inputParameters
* Type: struct
* Data Members: x_dim,y_dim,z_dim - Size of volumetric dataset along X,Y,and Z
*				var_start, dvar - Starting position, Delta
				                  Where var is in {x,y,z}
*               isovalue - Value for which iso-surface must be rendered
*               InputFile: Name of the file containing the volumetric dataset
*				OutputVertexFile: Name of the output file that contains
								  co-ordinates of vertices
				OutputIndexFile: Name of the output file that contains
								 vertex connectivity map
* Description: The structure <inputParameters> stores the dimension of input
*			   datset, and name of the input / output files.
*/
struct inputParameters {
	unsigned int x_dim, y_dim, z_dim;
	float x_start, y_start, z_start;
	float dx, dy, dz;
	float isovalue;
	std::string InputFile, OutputVertexFile, OutputIndexFile;
};
//-----------------------------------------------------------------------------
/*
* Function Name: StatusLog
* Return: void
* Arguments: const std::string - Contains the status message
*
* Description: The function accepts the message passed to it as a string and
*              prints it to the log file
*/
void StatusLog(const std::string);
//-----------------------------------------------------------------------------
/*
* Function Name: input_parameter_initialization
* Return: inputParameters
* Argument: const std::string - Contains the name of the file which has the
			input parameters
* Description: The functions initialises the input parameters
*/
inputParameters input_parameter_initialization(const std::string);
//-----------------------------------------------------------------------------
/*
* Function Name: inputRead
* Return: void
* Arguments: inputParameters: A structure containing the name of input
			 parameters
			 thrust::host_vector<float4> &: Host Vector 
* Description: The function reads the input data file and initialises a float4
			   vector (x,y,z,data) of size xdim * ydim * zdim

			   Row Major Order: [i][j][k] = i * ydim * zdim + j * zdim + k
*/
void inputRead(inputParameters, thrust::host_vector<float4> &);
//-----------------------------------------------------------------------------
/*
* Function Name: marchingCubes
* Return: void
* Arguments: inputParameters: A structure containing input parameters
             thrust::host_vector<float4>&: host_vector - [x,y,z,gridValue]
* Description: Function launches the kernel
*/
void marchingCubes(thrust::host_vector<float4>&, inputParameters);
//-----------------------------------------------------------------------------
/*
* Function Name: kernel
* Return: void
* Arguments: float - isovalue for the surface,
			 unsigned int - x,y,z dimension,
			 float4* - Device Vector (x,y,z,gridValue),
			 float3* - Vertex Bins,
			 int* - Traingle Counter
* Description: Performs Marching Cubes
*/
__global__ void kernel(float, unsigned int, unsigned int, unsigned int,
	float4*, float3*, int*);
//-----------------------------------------------------------------------------
/*
* Function Name: linearInterpolation
* Return: float3
* Arguments: float - Isovalue
			 float4 - Data of Voxel A
			 float4 - Data of Voxel B
* Description: Performs linear interpolation and finds position
*/
__device__ float3 linearInterpolation(float, float4, float4);
//-----------------------------------------------------------------------------
/*
* Function Name: writeOutputList
* Return: void
* Arguments: const int - totalSize  = xdim * ydim * zdim,
			 std::string - Vextex Output list filename,
			 std::string - Index Output list filename,
* Description: Prints the output lists to respective files
*/
void writeOutputList(const int, std::string, std::string);
//-----------------------------------------------------------------------------
/*
* Function Name: GetMemoryStatus
* Return: void
* Arguments: -
* Description: Prints the current available, used and total memory of the GPU
*/
void GetMemoryStatus();
//-----------------------------------------------------------------------------
/*****************************************************************************/
//-----------------------------------------------------------------------------
//Function Definitions
void StatusLog(const std::string status)
{
	//std::cout << status << std::endl;

	//Opening a file for output and seek to end before write
	std::ofstream LogFile("log.txt", std::ios::out | std::ios::app);

	//Checking if the file is open
	if (LogFile.is_open())
	{
		//Printing the status message to the file
		LogFile << status << std::endl;
	}
	else {
		//Printing error message to the screen when file fails to open
		std::cerr << "Error: Failed to open log file" << std::endl;
		exit(-1);
	}

	//Closing the output file stream
	LogFile.close();
}

inputParameters input_parameter_initialization(const std::string FileName)
{
	//Declaring an object of type <inputParameters>
	inputParameters data;

	//Opening a file for input
	std::ifstream ParamaterFile(FileName.c_str(), std::ios::in);

	//Checking if the file is open
	if (ParamaterFile.is_open())
	{
		
		//Reading the data from input file <FileName> and initialsing the
		//data members of the structure

		ParamaterFile >> data.x_dim >> data.y_dim >> data.z_dim
			>> data.x_start >> data.y_start >> data.z_start
			>> data.dx >> data.dy >> data.dz
			>> data.isovalue >> data.InputFile
			>> data.OutputVertexFile >> data.OutputIndexFile;

		//Updating status
		StatusLog("Initiliased the input parameters");
	}
	else
	{
		//Updating error
		StatusLog("Failed to open " + FileName);
		exit(-1);
	}

	//Closing the input file stream
	ParamaterFile.close();

	//Returning the structure
	return data;
}


void inputRead(inputParameters data, thrust::host_vector<float4> &host_vec)
{
	//Allocating Memory to X,Y,Z   
	thrust::host_vector<float> X(data.x_dim);
	thrust::host_vector<float> Y(data.y_dim);
	thrust::host_vector<float> Z(data.z_dim);


	//Generating X,Y,Z sequence
	thrust::sequence(thrust::host, 
					X.begin(), X.begin() + data.x_dim, data.x_start, data.dx);
	thrust::sequence(thrust::host,
					Y.begin(), Y.begin() + data.y_dim, data.y_start, data.dy);
	thrust::sequence(thrust::host,
					Z.begin(), Z.begin() + data.z_dim, data.z_start, data.dz);

	StatusLog("Generated Sequence for X,Y,Z");

	//Opening a file for reading the data
	std::ifstream inputFile(data.InputFile.c_str(), std::ios::in);

	//Checking if the file is open
	if (inputFile.is_open())
	{
		//Updating Status 
		StatusLog("Opened the file " + data.InputFile);

		//Temporary variables
		float temp;

		for (unsigned int x_index = 0; x_index < data.x_dim; x_index++) {
			for (unsigned int y_index = 0; y_index < data.y_dim; y_index++) {
				for (unsigned int z_index=0; z_index < data.z_dim; z_index++) {
					//Checking if data is read
					if (inputFile >> temp) {
						//Initialising host_vec
						host_vec.push_back(make_float4(
							X[x_index], Y[y_index], Z[z_index], temp));
					}
					else {
						StatusLog("Error: Failed to read " + data.InputFile);
						goto EXIT;
					}

				}
			}
		}

		StatusLog("Input initialised");
	}
	else {
		StatusLog("Failed to open:" + data.InputFile);
		exit(-1);
	}

EXIT:
	//Closing Stream
	inputFile.close();
	//Deleting memory allocated to X,Y,Z
	try {
		X.clear();
		X.shrink_to_fit();
		Y.clear();
		Y.shrink_to_fit();
		Z.clear();
		Z.shrink_to_fit();
	}
	catch (thrust::system_error e) {
		StatusLog(e.what());
		exit(-1);
	}
	return;
}

void GetMemoryStatus() {
	size_t available;
	size_t total;
	size_t used;

	cudaMemGetInfo(&available, &total);
	used = total - available;

	//Updating Memory Usage
	StatusLog("Used Memory: " + std::to_string(used));
	StatusLog("Available Memory: " + std::to_string(available));
	StatusLog("Total Memory: " + std::to_string(total));
	StatusLog("****************************************");

	return;
}

__device__ float3 linearInterpolation(float isovalue, float4 voxelA,
	float4 voxelB) {
	const float scale = (isovalue - voxelA.w) / (voxelB.w - voxelA.w);

	//Coordinates of position will be in float3
	float3 position;

	//Initialising position
	position.x = voxelA.x + scale * (voxelB.x - voxelA.x);
	position.y = voxelA.y + scale * (voxelB.y - voxelA.y);
	position.z = voxelA.z + scale * (voxelB.z - voxelA.z);

	return position;
}


__global__ void kernel(float isovalue, unsigned int xdim, unsigned int ydim,
	unsigned int zdim, float4* device, float3* vertex, int* triangle)
{

	//id has the voxel index
	uint3 id;

	//Initialising <id>
	id.x = blockIdx.x * blockDim.x + threadIdx.x;
	id.y = blockIdx.y * blockDim.y + threadIdx.y;
	id.z = blockIdx.z * blockDim.z + threadIdx.z;

	//There are xdim-1 * ydim -1 * zdim-1 voxel in a voxumteric data of size
	//[xdim,ydim,zdim]
	if (id.x < (xdim - 1) || id.y < (ydim - 1) || id.z << (zdim - 1)) {

		//Variables
		float4 voxels[8];
		float3 pos[12];
		int index[8];
		unsigned int cubeIndex = 0;
		float3 vertices[15];
		int numTriangles = 0;
		int numVertices = 0;

		//Getting the index of 8 vertices
		index[0] = xdim * (ydim * id.z + id.y) + id.x;
		index[1] = xdim * (ydim * id.z + id.y + 1) + id.x;
		index[2] = xdim * (ydim * id.z + id.y + 1) + id.x + 1;
		index[3] = xdim * (ydim * id.z + id.y) + id.x + 1;
		index[4] = xdim * (ydim * (id.z + 1) + id.y) + id.x;
		index[5] = xdim * (ydim * (id.z + 1) + id.y + 1) + id.x;
		index[6] = xdim * (ydim * (id.z + 1) + id.y + 1) + id.x + 1;
		index[7] = xdim * (ydim * (id.z + 1) + id.y) + id.x + 1;


		for (int i = 0; i < 8; ++i)
		{
			//Getting data of the vertices in this voxel
			voxels[i].x = device[index[i]].x;
			voxels[i].y = device[index[i]].y;
			voxels[i].z = device[index[i]].z;
			voxels[i].w = device[index[i]].w;

			//Comparing the grid value at 8 points
			if (voxels[i].w >= isovalue) {
				//cubeIndex is being left shifted based on the vertex
				cubeIndex |= (1 << i);
			}
		}


		//Getting edges
		unsigned int edges = edgeTable[cubeIndex];

		//Comparing edges with 12 bit by and operation and position coordinate
		if (edges == 0) {
			return;
		}
		if (edges & 1) {
			pos[0] = linearInterpolation(isovalue, voxels[0], voxels[1]);
		}
		if (edges & 2) {
			pos[1] = linearInterpolation(isovalue, voxels[1], voxels[2]);
		}
		if (edges & 4) {
			pos[2] = linearInterpolation(isovalue, voxels[2], voxels[3]);
		}
		if (edges & 8) {
			pos[3] = linearInterpolation(isovalue, voxels[3], voxels[0]);
		}
		if (edges & 16) {
			pos[4] = linearInterpolation(isovalue, voxels[4], voxels[5]);
		}
		if (edges & 32) {
			pos[5] = linearInterpolation(isovalue, voxels[5], voxels[6]);
		}
		if (edges & 64) {
			pos[6] = linearInterpolation(isovalue, voxels[6], voxels[7]);
		}
		if (edges & 128) {
			pos[7] = linearInterpolation(isovalue, voxels[7], voxels[4]);
		}
		if (edges & 256) {
			pos[8] = linearInterpolation(isovalue, voxels[0], voxels[4]);
		}
		if (edges & 512) {
			pos[9] = linearInterpolation(isovalue, voxels[1], voxels[5]);
		}
		if (edges & 1024) {
			pos[10] = linearInterpolation(isovalue, voxels[2], voxels[6]);
		}
		if (edges & 2048) {
			pos[11] = linearInterpolation(isovalue, voxels[3], voxels[7]);
		}

		for (int n = 0; n < 15; n += 3)
		{
			int edgeNumber = triTable[cubeIndex][n];
			if (edgeNumber < 0)
				break;

			vertices[numVertices++] = pos[edgeNumber];
			vertices[numVertices++] = pos[triTable[cubeIndex][n + 1]];
			vertices[numVertices++] = pos[triTable[cubeIndex][n + 2]];
			++numTriangles;
		}

		//Getting the number of triangles
		triangle[index[0]] = numTriangles;

		//Vertex List
		for (int n = 0; n < numVertices; ++n) {
			vertex[MAX_VERTEX * index[0] + n] = vertices[n];
		}
	}
	return;
}

void writeOutputList(const int totalSize, std::string IndexFile, std::string
	VertexFile)
{
	//Creating Output Streams
	std::fstream Output[4];

	//Opening Files
	Output[0].open(IndexFile.c_str(), std::ios::out);
	Output[1].open(VertexFile.c_str()
		+ std::to_string(1) + ".dat", std::ios::out);

	Output[2].open(VertexFile.c_str()
		+ std::to_string(2) + ".dat", std::ios::out);

	Output[3].open(VertexFile.c_str()
		+ std::to_string(3) + ".dat", std::ios::out);

	//Printing Counter to file 
	if (Output[0].is_open()) {
		StatusLog("Writting Index File");
		for (int i = 0; i < totalSize; i++) {
			Output[0] << counter[i] << "\t";
		}
	}
	else
	{
		StatusLog("Error: Failed to open output files:" + IndexFile);
		exit(-1);
	}

	delete[]counter;

	Output[0].close();

	//Printing Verxtex list
	if (Output[1].is_open() && Output[2].is_open() && Output[3].is_open()) 
	{
		StatusLog("Writting vertex files");
		for (int i = 0; i < totalSize; i++) {
			Output[1] << bin[i * MAX_VERTICES].x << "\t";
			Output[2] << bin[i * MAX_VERTICES].y << "\t";
			Output[3] << bin[i * MAX_VERTICES].z << "\t";
		}
	}
	else if(Output[1].is_open() != 1)
	{
		StatusLog("Error: Failed to open " + VertexFile + std::to_string(1));
		exit(-1);
	}
	else if(Output[2].is_open() != 1)
	{
		StatusLog("Error: Failed to open " + VertexFile + std::to_string(2));
		exit(-1);
	}
	else
	{
		StatusLog("Error: Failed to open " + VertexFile + std::to_string(3));
		exit(-1);
	}
	
	delete[]bin;

	//Closing streams
	Output[1].close();
	Output[2].close();
	Output[3].close();

	return;
}

void marchingCubes(thrust::host_vector<float4>& host_vec,inputParameters data){

	//Size of the grid
	int totalSize = int(data.x_dim * data.y_dim * data.z_dim);

	//4 x 4 x 4 block dimension
	dim3 blockSize(4, 4, 4);

	//Initialising grid dimension
	dim3 gridSize((data.x_dim + blockSize.x - 1) / blockSize.x,
		(data.y_dim + blockSize.y - 1) / blockSize.y,
		(data.z_dim + blockSize.z - 1) / blockSize.z);

	//Updating log file 
	StatusLog("Invoking Kernel");
	
	//Updating Memory Status
	GetMemoryStatus();

	//Calling cuda kernel cudaEvent_t start, stop;
	cudaEvent_t start, stop;
	float elapsedTime;

	//Creating Start and Stop events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	// Host to Device Memory Copy
	thrust::device_vector<float4> device_vec = host_vec;
	
	//Casting a raw pointer for the device vector
	float4 *device_pointer = thrust::raw_pointer_cast(device_vec.data());

	//Output in the GPU will stored in vertex and triangle
	float3* vertex;
	int* triangle;
		
	cudaMalloc(&vertex, sizeof(float3) * totalSize * MAX_VERTICES);
	cudaMemset(vertex, 0, sizeof(float3) * totalSize * MAX_VERTICES);
	cudaMalloc(&triangle, sizeof(int) * totalSize);
	cudaMemset(triangle, 0, sizeof(int) * totalSize);
	

	//Invoking Kernel
	kernel <<<gridSize, blockSize >>> (data.isovalue, data.x_dim, data.y_dim,
		data.z_dim, device_pointer, vertex, triangle);

	
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		std::cout << cudaGetErrorString(error) << std::endl;
		exit(-1);
	}
	
	//Memcopy - Device to Host
	bin = (float3*)malloc(sizeof(float3) * totalSize * MAX_VERTICES);
	cudaMemcpy(vertex, bin, sizeof(float3) * totalSize * MAX_VERTICES, 
													   cudaMemcpyDeviceToHost);

	counter = (int*)malloc(sizeof(int) * totalSize);
	cudaMemcpy(counter, counter, sizeof(int) * totalSize,
													   cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	StatusLog("Returned to host");
	StatusLog("Elapsed Time: " + std::to_string(elapsedTime));
	
	//Destroying events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//Clearing the Host Memory
	try {
		host_vec.clear();
		host_vec.shrink_to_fit();
	}
	catch (thrust::system_error e) {
		StatusLog(e.what());
		exit(-1);
	}
	
	GetMemoryStatus();

	//Clearing Device Memory
	cudaFree(triangle);
	cudaFree(vertex);
	cudaError_t MemoryError = cudaGetLastError();
	
	if (cudaSuccess != MemoryError)
	{
		StatusLog(cudaGetErrorString(MemoryError));
		exit(-1);
	}
	
	try {
		device_vec.clear();
		device_vec.shrink_to_fit();
	}
	catch (thrust::system_error e)
	{
		StatusLog(e.what());
		exit(-1);
	}
	
	GetMemoryStatus();
	
	//Invoking Function to Create and Print Index List and Vertex List
	StatusLog("Creating Lists");

	//Writting Output
	writeOutputList(totalSize, data.OutputIndexFile, data.OutputVertexFile);
	
	return;
}

//-----------------------------------------------------------------------------
// Main function
//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
	
	//Declaring and defining an object to struct <inputParameters>
	inputParameters data = input_parameter_initialization(std::string(argv[1]));

	thrust::host_vector<float4> host_vec;

	//Declaring and intialising the volumetric data
	inputRead(data, host_vec);

	//Calling marchingCubes to invoke kernel
	marchingCubes(host_vec, data);

	//Clearing Device Memory

	StatusLog("Execution Successfully");
	//Execution Successful

	return (0);
}

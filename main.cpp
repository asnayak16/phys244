/*****************************************************************************
  Last Update: 14th June, 2017
  -----------------------------------------------------------------------------
  Author: 		Prashant Singh (Marching Cube Implementation)
  			Ganesh Ranganath Chandrasekar Iyer (Initial skeletal for input)
			Ashwin Nayak	
  Objective: 	Implements the Marching Cube's Algorithm using OpenMP.
  Functions: 	input_parameter_initialization,
 				inputRead,releaseHostData
  Input: 		Name of the file which has the input parameters in command line
		 		iso_data.dat: File which has the volumetric data
*****************************************************************************/
//-----------------------------------------------------------------------------
// Including Header Files

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "MCTable.h"
#include <chrono>

using namespace std;

#define ABS(x) (x < 0 ? -(x) : (x))

//-----------------------------------------------------------------------------
//Declaring Structures

// Vertex Structure
typedef struct {
   float x,y,z;
} XYZ;

// Cube structure
typedef struct {
   XYZ p[8];
   XYZ n[8];
   float val[8];
} CUBE;

// Triangle Struct
typedef struct {
   XYZ p[3];         /* Vertices */
   XYZ c;            /* Centroid */
   XYZ n[3];         /* Normal   */
} TRIANGLE;

// Input params
struct inputParameters {
	unsigned int x_dim, y_dim, z_dim;
	float x_start, x_end, y_start, y_end, z_start, z_end;
	float dx, dy, dz;
	float isolevel;
	string InputFile;
	string OutputFile;
};


//-----------------------------------------------------------------------------
// Function Declarations

int PolygoniseCube(CUBE,float,TRIANGLE *);
//
XYZ VertexInterp(float,XYZ,XYZ,float,float);
//
inputParameters input_parameter_initialization(const string);
//
float*** inputRead(inputParameters, float *X, float *Y, float *Z,
		float & max_value, float & min_value);
//
void releaseHostData(float ***, const int, const int, const int);

//=============================================================================
// MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
	using std::chrono::high_resolution_clock;
	using std::chrono::milliseconds;
	//declaring variables for data analysis
	float max_value = 0, min_value = 0, isolevel = 0;
	//static int x_count, y_count, z_count;
	//x_count = y_count = z_count = 0;

	unsigned int i,j,k,l,n;
	CUBE cube;
	TRIANGLE triangles[10];
	TRIANGLE *tri = NULL;
	unsigned int ntri = 0;
	FILE *fptr;
	FILE *fptrv;

	//Declaring and defining an object to struct <inputParameters>
	inputParameters data= input_parameter_initialization(string(argv[1]));

	//extracting isolevel from the data
	isolevel = data.isolevel;

	//Declaring & initializing the X, Y and Z matrices.
	float * X = NULL, *Y = NULL, *Z = NULL;
	X = new float[data.x_dim];
	Y = new float[data.y_dim];
	Z = new float[data.z_dim];

	//Declaring and initializing the host data
	float*** H_data = inputRead(data, X, Y, Z, max_value, min_value);

	//Deleting the memory allocated
	//releaseHostData(H_data, data.x_dim, data.y_dim, data.z_dim);

	fprintf(stderr,"Volumetric data range: %f -> %f\n",min_value, max_value);

	// Polygonise the cube
	fprintf(stderr,"Polygonising data ...\n");
	
	// Start Stopwatch
	auto t0 = high_resolution_clock::now();

	#pragma omp parallel for schedule(static) \
	 	    	private(i,j,k,l,n,cube,triangles) shared(H_data,tri,ntri)
	for (i=0;i< data.x_dim-1;i++) {
		for (j=0;j<data.y_dim-1;j++) {
			for (k=0;k<data.z_dim-1;k++) {
				cube.p[0].x = X[i];
				cube.p[0].y = Y[j];
				cube.p[0].z = Z[k];
				cube.val[0] = H_data[i][j][k];
				cube.p[1].x = X[i+1];
				cube.p[1].y = Y[j];
				cube.p[1].z = Z[k];
				cube.val[1] = H_data[i+1][j][k];
				cube.p[2].x = X[i+1];
				cube.p[2].y = Y[j+1];
				cube.p[2].z = Z[k];
				cube.val[2] = H_data[i+1][j+1][k];
				cube.p[3].x = X[i];
				cube.p[3].y = Y[j+1];
				cube.p[3].z = Z[k];
				cube.val[3] = H_data[i][j+1][k];
				cube.p[4].x = X[i];
				cube.p[4].y = Y[j];
				cube.p[4].z = Z[k+1];
				cube.val[4] = H_data[i][j][k+1];
				cube.p[5].x = X[i+1];
				cube.p[5].y = Y[j];
				cube.p[5].z = Z[k+1];
				cube.val[5] = H_data[i+1][j][k+1];
				cube.p[6].x = X[i+1];
				cube.p[6].y = Y[j+1];
				cube.p[6].z = Z[k+1];
				cube.val[6] = H_data[i+1][j+1][k+1];
				cube.p[7].x = X[i];
				cube.p[7].y = Y[j+1];
				cube.p[7].z = Z[k+1];
				cube.val[7] = H_data[i][j+1][k+1];
				n = PolygoniseCube(cube,isolevel,triangles);
				#pragma omp critical {
					tri = (TRIANGLE*) realloc(tri,(ntri+n)*sizeof(TRIANGLE));
					for (l=0;l<n;l++) tri[ntri+l] = triangles[l];
					ntri += n;
				}
			}
		}
	}
	
	

	fprintf(stderr,"Total of %d triangles\n",ntri);

	// Writing the triangles to the file
	fprintf(stderr,"Writing triangles & vertices ...\n");
	if ((fptr = fopen("vertices.dat","w")) == NULL) {
		fprintf(stderr,"Failed to open indices file\n");
		exit(-1);
	}
	
	for (i=0;i<ntri;i++) {
      //fprintf(fptrv,"%d \n", ntri);
      for (k=0;k<3;k++)  {
         fprintf(fptr,"%f %f %f \n",tri[i].p[k].x,tri[i].p[k].y,tri[i].p[k].z);
      }
		//fprintf(fptr,"\n"); // spacing between triangle co-ordinates
	}

	//Calculating indices and serializing it.
	if ((fptrv = fopen("indices.dat","w")) == NULL) {
		fprintf(stderr,"Failed to open vertices file\n");
		exit(-1);
	}

	int totalSize = data.x_dim * data.y_dim * data.z_dim;
	unsigned int* indices[3];

	for (int i = 0; i < 3; ++i)
	    {
	        indices[i] = (unsigned int*)malloc(sizeof(unsigned int) * ntri);
	    }

    int tIdx = 0, vIdx = 0;


        for (int c = 0; c < ntri; ++c)
        {
            for (int v = 0; v < 3; ++v)
            {
                indices[v][tIdx] = 3 * tIdx + v + 1;
                fprintf(fptrv,"%d ", indices[v][tIdx]);
                ++vIdx;
            }
            fprintf(fptrv,"\n");
            ++tIdx;
        }

    //closing file streams
	fclose(fptr);
	fclose(fptrv);

auto t1 = high_resolution_clock::now();
	milliseconds total_ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout << "Time Taken : " << total_ms.count() << "ms\n";


	exit(0);
	return 1;
}

// Input Params from file
inputParameters input_parameter_initialization(const string FileName)
{
	//Declaring an object of type <inputParameters>
	inputParameters data;

	//Opening a file for input
	ifstream ParamaterFile(FileName.c_str(), ios::in);

	//Checking if the file is open
	if (ParamaterFile.is_open())
	{
	     // Reading the data from input file <FileName> and initialsing the 
		 // data members of the structure	 
		ParamaterFile >> data.x_dim >> data.y_dim >> data.z_dim
			>> data.x_start >> data.x_end
			>> data.y_start >> data.y_end
			>> data.z_start >> data.z_end
			>> data.dx >> data.dy >> data.dz
			>> data.isolevel
			>> data.InputFile >> data.OutputFile;
	}
	else
	{
		//Updating error
		exit(-1);
	}

	//Closing the input file stream
	ParamaterFile.close();

	//Returning the structure
	return data;
}

/* Read Field Dataset */
float*** inputRead(inputParameters data, float *X, float *Y, float *Z, float & max_value, float & min_value)
{
	//declaring variable to find max, min value
	float c = 0;

	// Allocating memory to a 3D array gridValue
	float *** gridValue = new float**[data.x_dim];
	for (unsigned int x_index = 0; x_index < data.x_dim; ++x_index) {
		gridValue[x_index] = new float*[data.y_dim];
		for (unsigned int y_index = 0; y_index < data.y_dim; ++y_index) {
			gridValue[x_index][y_index] = new float[data.z_dim];
		}
	}

	for (unsigned int x_index = 0; x_index < data.x_dim; x_index++)
		X[x_index] = data.x_start + (x_index+1)*data.dx;

	for (unsigned int y_index = 0; y_index < data.y_dim; y_index++)
		Y[y_index] = data.y_start + (y_index+1)*data.dy;

	for (unsigned int z_index = 0; z_index < data.z_dim; z_index++)
		Z[z_index] = data.z_start + (z_index+1)*data.dz;

	//Opening a file for reading the data
	ifstream inputFile(data.InputFile.c_str(), ios::in);
	
	//Checking if the file is open
	if (inputFile.is_open()) {
		//Defining the data members of the structure
		for (unsigned int x_index = 0; x_index < data.x_dim; ++x_index) {
			for (unsigned int y_index = 0; y_index < data.y_dim; ++y_index) {
				for (unsigned int z_index = 0; z_index < data.z_dim; ++z_index) {
					inputFile >> gridValue[x_index][y_index][z_index];
					c = gridValue[x_index][y_index][z_index];
					if (c > max_value)
						max_value = c;
					if (c < min_value)
						min_value = c;
				}
			}
		}
	}
	//Returning a pointer
	return gridValue;
}

// Release Data to free Memory 
void releaseHostData(float ***h_data,
	const int x_dim, const int y_dim, const int z_dim) {
	//Deleting the memory allocated to the structure 
	for (int x_index = 0; x_index < x_dim; ++x_index) {
		for (int y_index = 0; y_index < y_dim; ++y_index) {
			delete[] h_data[x_index][y_index];
		}
		delete[] h_data[x_index];
	}
	delete[] h_data;
}

/***************************************************************************
% Description : Given a grid cell and an isolevel, calculate the triangular
%				facets requied to represent the isosurface through the cell.
%				Return the number of triangular facets, the array "triangles"
%				will be loaded up with the vertices at most 5 triangular facets.
%				0 will be returned if the grid cell is either totally above
%				of totally below the isolevel.
% Note :		The following method is adapted from Paul Baurke's
				implementation.
 ***************************************************************************/
int PolygoniseCube(CUBE g,float iso,TRIANGLE *tri)
{
   int i,ntri = 0;
   int cubeindex;
   XYZ vertlist[12];

   /*
      Determine the index into the edge table which
      tells us which vertices are inside of the surface
   */
   cubeindex = 0;
   if (g.val[0] < iso) cubeindex |= 1;
   if (g.val[1] < iso) cubeindex |= 2;
   if (g.val[2] < iso) cubeindex |= 4;
   if (g.val[3] < iso) cubeindex |= 8;
   if (g.val[4] < iso) cubeindex |= 16;
   if (g.val[5] < iso) cubeindex |= 32;
   if (g.val[6] < iso) cubeindex |= 64;
   if (g.val[7] < iso) cubeindex |= 128;

   /* Cube is entirely in/out of the surface */
   if (edgeTable[cubeindex] == 0)
      return(0);

   /* Find the vertices where the surface intersects the cube 
   according to the configuration obtained from edgetable
   */
   if (edgeTable[cubeindex] & 1) {
      vertlist[0] = VertexInterp(iso,g.p[0],g.p[1],g.val[0],g.val[1]);
   }
   if (edgeTable[cubeindex] & 2) {
      vertlist[1] = VertexInterp(iso,g.p[1],g.p[2],g.val[1],g.val[2]);
   }
   if (edgeTable[cubeindex] & 4) {
      vertlist[2] = VertexInterp(iso,g.p[2],g.p[3],g.val[2],g.val[3]);
   }
   if (edgeTable[cubeindex] & 8) {
      vertlist[3] = VertexInterp(iso,g.p[3],g.p[0],g.val[3],g.val[0]);
   }
   if (edgeTable[cubeindex] & 16) {
      vertlist[4] = VertexInterp(iso,g.p[4],g.p[5],g.val[4],g.val[5]);
   }
   if (edgeTable[cubeindex] & 32) {
      vertlist[5] = VertexInterp(iso,g.p[5],g.p[6],g.val[5],g.val[6]);
   }
   if (edgeTable[cubeindex] & 64) {
      vertlist[6] = VertexInterp(iso,g.p[6],g.p[7],g.val[6],g.val[7]);
   }
   if (edgeTable[cubeindex] & 128) {
      vertlist[7] = VertexInterp(iso,g.p[7],g.p[4],g.val[7],g.val[4]);
   }
   if (edgeTable[cubeindex] & 256) {
      vertlist[8] = VertexInterp(iso,g.p[0],g.p[4],g.val[0],g.val[4]);
   }
   if (edgeTable[cubeindex] & 512) {
      vertlist[9] = VertexInterp(iso,g.p[1],g.p[5],g.val[1],g.val[5]);
   }
   if (edgeTable[cubeindex] & 1024) {
      vertlist[10] = VertexInterp(iso,g.p[2],g.p[6],g.val[2],g.val[6]);
   }
   if (edgeTable[cubeindex] & 2048) {
      vertlist[11] = VertexInterp(iso,g.p[3],g.p[7],g.val[3],g.val[7]);
   }

   /* Triangulation */
   for (i=0;triTable[cubeindex][i]!=-1;i+=3) {
      tri[ntri].p[0] = vertlist[triTable[cubeindex][i  ]];
      tri[ntri].p[1] = vertlist[triTable[cubeindex][i+1]];
      tri[ntri].p[2] = vertlist[triTable[cubeindex][i+2]];
      ntri++;
   }
   return(ntri);
}

/* Function to linearly interpolate between two vertices */
XYZ VertexInterp(float isolevel,XYZ p1,XYZ p2,float valp1,float valp2)
{
   float mu;
   XYZ p;

   if (ABS(isolevel-valp1) < 0.00001)
      return(p1);
   if (ABS(isolevel-valp2) < 0.00001)
      return(p2);
   if (ABS(valp1-valp2) < 0.00001)
      return(p1);
   mu = (isolevel - valp1) / (valp2 - valp1);
   p.x = p1.x + mu * (p2.x - p1.x);
   p.y = p1.y + mu * (p2.y - p1.y);
   p.z = p1.z + mu * (p2.z - p1.z);

   return(p);
}

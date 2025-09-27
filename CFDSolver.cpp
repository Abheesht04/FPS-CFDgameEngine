#define NOMINMAX
#include <windows.h>
#include <iostream>
#include <string>
#include <limits>
#include <vector>
#include <fstream>

#include "Mesher.h"

using namespace std;

// Simple function to find first .vtk file in current directory
std::string FindVTKFile()
{
    WIN32_FIND_DATAW findData = {};
    HANDLE hFind = FindFirstFileW(L"*.vtk", &findData);
    if (hFind == INVALID_HANDLE_VALUE)
        return "";

    // Convert WCHAR* to std::string UTF-8
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, findData.cFileName, -1,
        nullptr, 0, nullptr, nullptr);
    std::string filename(size_needed - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, findData.cFileName, -1,
        &filename[0], size_needed, nullptr, nullptr);

    FindClose(hFind);
    return filename;
}

int RunMeshGeneration()
{
    string stlFilename;
    double grid_mm;

    cout << "Enter STL filename: ";
    getline(cin, stlFilename);

    cout << "Enter grid spacing in mm: ";
    cin >> grid_mm;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    double dx = grid_mm / 1000.0;

    Mesh mesh;
    mesh.loadSTL(stlFilename);
    mesh.repairDeduplicate();
    mesh.computeBoundingBox();
    mesh.fixBoundaryOrientation();
    mesh.patchifySurface(15);
    mesh.remeshSurfacePatches();
    mesh.generateGridPoints(dx);
    mesh.filterGridPoints();
    mesh.superTetra();
    mesh.delaunayTetrahedralize();
    mesh.writeVTK("mesh.vtk");

    cout << "Mesh generated and saved to mesh.vtk" << endl;

    return 0;
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int nCmdShow)
{
    // Allocate console for this GUI app to do console I/O
    if (!AllocConsole())
        return -1;

    FILE* conin = nullptr;
    FILE* conout = nullptr;
    freopen_s(&conin, "CONIN$", "r", stdin);
    freopen_s(&conout, "CONOUT$", "w", stdout);
    freopen_s(&conout, "CONOUT$", "w", stderr);

    // Run console mesh generation
    if (RunMeshGeneration() != 0) {
        cout << "Mesh generation failed. Press Enter to exit." << endl;
        cin.get();
        FreeConsole();
        return -1;
    }

    // After mesh generation, find the .vtk file
    string vtkFileName = FindVTKFile();
    if (vtkFileName.empty()) {
        cout << "No .vtk file found in current directory." << endl;
    }
    else {
        cout << "Found mesh file: " << vtkFileName << endl;
    }

    cout << "Press Enter to exit." << endl;
    cin.get();

    FreeConsole();
    return 0;
}

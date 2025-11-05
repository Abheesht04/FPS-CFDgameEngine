#pragma once
#include<cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include<iostream>

// -------------------- Basic Types --------------------

struct Point3D {
    double x{ 0.0 }, y{ 0.0 }, z{ 0.0 };
    
};


struct BoundingBox {
    Point3D min;
    Point3D max;
};

struct Cell {
    std::vector<int> nodeIds; // indices into points
    int type{ -1 };           // VTK cell type code
};

struct Face {
    std::vector<int> nodeIds; // original node order
    int leftCell{ -1 };
    int rightCell{ -1 };
    Point3D center;
    Point3D normal;
    double area{ 0.0 };
};

// -------------------- Hash for Face --------------------

struct FaceKey {
    std::vector<int> nodes; // always sorted for uniqueness
    bool operator==(const FaceKey& other) const;
};

struct FaceKeyHash {
    std::size_t operator()(const FaceKey& key) const;
};

// -------------------- Post-processor Mesh --------------------

class postproc
{
public:
    postproc() = default;

    // Load a legacy VTK file into this mesh
    void loadFromVTK(const std::string& filename);

    // Accessors
    [[nodiscard]] const std::vector<Point3D>& getPoints() const noexcept;
    [[nodiscard]] const std::vector<Cell>& getCells()  const noexcept;
    [[nodiscard]] const std::vector<Face>& getFaces()  const noexcept;

    std::size_t pointCount() const noexcept;
    std::size_t cellCount()  const noexcept;
    std::size_t faceCount()  const noexcept;

    BoundingBox boundingBox() const;
    Point3D     centroid()    const;

    // Center the mesh around origin
    void centerMesh();

    void printOrientation() const;

    // -------------------- Face Connectivity --------------------
    void buildFaceConnectivity();

   

private:
    std::vector<Point3D> points_;
    std::vector<Cell> cells_;
    std::vector<Face> faces_;
    std::unordered_map<FaceKey, int, FaceKeyHash> faceMap_;
    std::vector<std::vector<int>> nodeToFace_;

    // -------------------- Helpers --------------------
    BoundingBox computeBoundingBox(const std::vector<Point3D>& pts) const;
    Point3D computeCentroid(const std::vector<Point3D>& pts) const;
    void translateMesh(std::vector<Point3D>& pts, const Point3D& shift);
    void checkOrientation(const BoundingBox& box) const;

    std::vector<std::vector<int>> getCellFaces(const Cell& cell) const;
};

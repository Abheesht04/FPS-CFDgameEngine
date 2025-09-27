#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <tuple>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>       // for std::queue
#include <set>         // for std::set
#include <unordered_set>
#include <cmath>       // for std::fabs, std::sqrt, M_PI (but M_PI may need defining)
#include <random>      // for std::mt19937, std::uniform_int_distribution
#include <array>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


class Vector3D {
public:
    double x, y, z;
    Vector3D(double x_val = 0.0, double y_val = 0.0, double z_val = 0.0);
    bool operator==(const Vector3D& other) const;
    Vector3D operator+(const Vector3D& other) const;
    Vector3D operator-(const Vector3D& other) const;
    Vector3D operator*(double scalar) const;
    double dot(const Vector3D& other) const;
    Vector3D cross(const Vector3D& other) const;
    double magnitude() const;
    Vector3D normalize() const;
    Vector3D operator/(double scalar) const {
        return Vector3D(x / scalar, y / scalar, z / scalar);
    }

    Vector3D& operator/=(double scalar) {
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }

    


    double& operator[](int i) {
        if (i == 0) return x;
        else if (i == 1) return y;
        else return z;
    }

    const double& operator[](int i) const {
        if (i == 0) return x;
        else if (i == 1) return y;
        else return z;
    }
};

struct VecHash {
    size_t operator()(const Vector3D& v) const;
};

struct VecEq {
    bool operator()(const Vector3D& a, const Vector3D& b) const;
};

class Triangle {
public:
    int v0, v1, v2;
    Triangle(int a, int b, int c);
};



class Tetrahedron {
public:
    int v0, v1, v2, v3;
    Tetrahedron(int a, int b, int c, int d);
};

struct Face {
    int v[3];
    Face(int a, int b, int c);
    bool operator==(const Face& other) const;
};

struct FaceHash {
    size_t operator()(const Face& f) const;
};

struct CircumSphere {
    Vector3D center;
    double radius;
};


struct Patch {
    std::vector<int> triangles;  // triangle indices
    Vector3D normal;             // average patch normal
    Vector3D centroid;           // patch centroid

    // Covariance matrix for plane fitting using plain arrays
    double cov[3][3];            // 3x3 covariance
    Vector3D uAxis, vAxis;       // 2D plane axes
};

// Node for kd-tree
struct KDNode {
    int index;             // index of element (tet ID in your case)
    double point[3];       // position (circumcenter or vertex)
    KDNode* left, * right;

    KDNode(int idx, const double* p);
};

// Main kd-tree class
class KdTree {
public:
    KDNode* root;

    KdTree();
    ~KdTree();

    void buildTree(const std::vector<std::pair<int, std::array<double, 3>>>& elems);

    std::vector<int> query(const double* target, double radius);

private:
    KDNode* build(std::vector<std::pair<int, std::array<double, 3>>>& elems, int depth = 0);
    void clear(KDNode* node);
    void radiusSearch(KDNode* node, const double* target, double radius, int depth,
        std::vector<int>& results);
    double distance(const double* a, const double* b);
};

// --- Custom hash for 3D integer tuple ---
struct Tuple3Hash {
    size_t operator()(const std::tuple<int64_t, int64_t, int64_t>& t) const noexcept {
        return std::hash<int64_t>{}(std::get<0>(t)) ^
            (std::hash<int64_t>{}(std::get<1>(t)) << 1) ^
            (std::hash<int64_t>{}(std::get<2>(t)) << 2);
    }
};

struct Tuple3Eq {
    bool operator()(const std::tuple<int64_t, int64_t, int64_t>& a,
        const std::tuple<int64_t, int64_t, int64_t>& b) const noexcept {
        return std::get<0>(a) == std::get<0>(b) &&
            std::get<1>(a) == std::get<1>(b) &&
            std::get<2>(a) == std::get<2>(b);
    }
};


inline bool solve3x3(const std::array<std::array<double, 3>, 3>& M, const std::array<double, 3>& b, Vector3D& x) {
    double det =
        M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
        M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
        M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

    if (std::fabs(det) < 1e-15) return false; // Singular or nearly degenerate

    double detX =
        b[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
        M[0][1] * (b[1] * M[2][2] - M[1][2] * b[2]) +
        M[0][2] * (b[1] * M[2][1] - M[1][1] * b[2]);

    double detY =
        M[0][0] * (b[1] * M[2][2] - M[1][2] * b[2]) -
        b[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
        M[0][2] * (M[1][0] * b[2] - b[1] * M[2][0]);

    double detZ =
        M[0][0] * (M[1][1] * b[2] - b[1] * M[2][1]) -
        M[0][1] * (M[1][0] * b[2] - b[1] * M[2][0]) +
        b[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

    x = Vector3D(detX / det, detY / det, detZ / det);
    return true;
}


struct pair_hash {
    template<class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

struct AABB {
    Vector3D min;
    Vector3D max;

    AABB()
        : min(Vector3D(1e18, 1e18, 1e18)),
        max(Vector3D(-1e18, -1e18, -1e18)) {}

    explicit AABB(const Vector3D& p)
        : min(p), max(p) {}

    void reset() {
        min = Vector3D(1e18, 1e18, 1e18);
        max = Vector3D(-1e18, -1e18, -1e18);
    }

    // Grow by a point (mm)
    void extend(const Vector3D& p) {
        if (p.x < min.x) min.x = p.x; if (p.x > max.x) max.x = p.x;
        if (p.y < min.y) min.y = p.y; if (p.y > max.y) max.y = p.y;
        if (p.z < min.z) min.z = p.z; if (p.z > max.z) max.z = p.z;
    }

    // Grow by another box (mm)
    void extend(const AABB& b) {
        extend(b.min);
        extend(b.max);
    }

    // Robust ray-box slab test (mm)
    bool intersect(const Vector3D& orig,
        const Vector3D& invDir,
        const std::array<int, 3>& /*unused*/) const {
        double t1 = (min.x - orig.x) * invDir.x;
        double t2 = (max.x - orig.x) * invDir.x;
        double tmin = std::min(t1, t2);
        double tmax = std::max(t1, t2);

        t1 = (min.y - orig.y) * invDir.y;
        t2 = (max.y - orig.y) * invDir.y;
        tmin = std::max(tmin, std::min(t1, t2));
        tmax = std::min(tmax, std::max(t1, t2));

        t1 = (min.z - orig.z) * invDir.z;
        t2 = (max.z - orig.z) * invDir.z;
        tmin = std::max(tmin, std::min(t1, t2));
        tmax = std::min(tmax, std::max(t1, t2));

        return tmax >= std::max(0.0, tmin);
    }
};


struct BVHNode {
    AABB bounds;
    int left = -1, right = -1;
    int start = 0, range = 0; // range of triangle indices for leaf
};




class Mesh {
public:
    std::vector<Vector3D> points;
    std::vector<Triangle> triangles;
    std::vector<Tetrahedron> tets;
    std::vector<Vector3D> insidePoints;
    std::vector<Vector3D> vertexNormals;
    Vector3D minBounds, maxBounds;
    static std::unordered_map<std::tuple<long long, long long, long long>, int> vertexMap;
    std::vector<Patch> patches;
    std::vector<BVHNode> bvhNodes;
    std::vector<int> triIndices;
    void loadSTL(const std::string& filename);
    void loadBinarySTL(std::ifstream& in);
    void loadAsciiSTL(std::ifstream& in);
    int addVertex(const Vector3D& v);
    int rootBVHNode;
    void computeNormals();
    void computeBoundingBox();
    void generateGridPoints(double dx);
    bool pointInsideMesh(const Vector3D& p,int numRays=6);
    void filterGridPoints();
    CircumSphere computeCircumsphere(const Vector3D& A, const Vector3D& B, const Vector3D& C, const Vector3D& D);
    bool pointInCircumsphere(const Vector3D& p, const Tetrahedron& tet);
    void superTetra();
    void delaunayTetrahedralize();
    void writeVTK(const std::string& filename);
    void repairDeduplicate();
    void fixBoundaryOrientation();
    void patchifySurface(double normalThresholdDeg = 10.0);
    void remeshSurfacePatches();
    void remeshPatch(Patch& patch);
    void logTriangleNormals();
    bool isWatertight();
    int buildBVH(int start, int end);
    bool intersectTriangle(const Vector3D& orig, const Vector3D& dir, const Triangle& tri, double& t);
    int traverseBVH(int nodeIdx, const Vector3D& orig, const Vector3D& dir, const Vector3D& invDir, const std::array<int, 3>& dirIsNeg);

    Vector3D projectTo2D(const Vector3D& v, const Patch& patch, double& u, double& w);
    Vector3D projectTo3D(double u, double w, const Patch& patch);

};

#include "Mesher.h"
#include <iomanip>
#include <random>
#include<unordered_set>
#include <omp.h> // Include the OpenMP header
#include <array>
#include <unordered_map>
#include <chrono>

// --------------------------------------------------
// Vector3D implementation
// --------------------------------------------------
Vector3D::Vector3D(double x_val, double y_val, double z_val)
    : x(x_val), y(y_val), z(z_val) {}

bool Vector3D::operator==(const Vector3D& other) const {
    return std::fabs(x - other.x) < 1e-9 &&
           std::fabs(y - other.y) < 1e-9 &&
           std::fabs(z - other.z) < 1e-9;
}

Vector3D Vector3D::operator+(const Vector3D& other) const {
    return Vector3D(x + other.x, y + other.y, z + other.z);
}
Vector3D Vector3D::operator-(const Vector3D& other) const {
    return Vector3D(x - other.x, y - other.y, z - other.z);
}
Vector3D Vector3D::operator*(double scalar) const {
    return Vector3D(x * scalar, y * scalar, z * scalar);
}
double Vector3D::dot(const Vector3D& other) const {
    return x * other.x + y * other.y + z * other.z;
}
Vector3D Vector3D::cross(const Vector3D& other) const {
    return Vector3D(
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x
    );
}
double Vector3D::magnitude() const {
    return std::sqrt(x * x + y * y + z * z);
}
Vector3D Vector3D::normalize() const {
    double m = magnitude();
    if (m < 1e-12) return *this;
    return Vector3D(x / m, y / m, z / m);
}

Vector3D Mesh::projectTo2D(const Vector3D& v, const Patch& patch, double& u, double& w) {
    Vector3D diff = v - patch.centroid;
    u = diff.dot(patch.uAxis);
    w = diff.dot(patch.vAxis);
    return Vector3D(u, w, 0);
}

Vector3D Mesh::projectTo3D(double u, double w, const Patch& patch) {
    return patch.centroid + patch.uAxis * u + patch.vAxis * w;
}

namespace std {
    template <>
    struct hash<std::tuple<long long, long long, long long>> {
        size_t operator()(const std::tuple<long long, long long, long long>& t) const noexcept {
            auto h1 = std::hash<long long>{}(std::get<0>(t));
            auto h2 = std::hash<long long>{}(std::get<1>(t));
            auto h3 = std::hash<long long>{}(std::get<2>(t));
            // combine hashes
            size_t seed = h1;
            seed ^= h2 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}

std::unordered_map<std::tuple<long long, long long, long long>, int> Mesh::vertexMap;


// Rounding helper
static std::tuple<long long, long long, long long> toKey(const Vector3D& v) {
    const double scale = 1e9; // precision (1e-9 tolerance)
    return {
        static_cast<long long>(std::llround(v.x * scale)),
        static_cast<long long>(std::llround(v.y * scale)),
        static_cast<long long>(std::llround(v.z * scale))
    };
}





// --------------------------------------------------
// Hashers and helpers
// --------------------------------------------------
size_t VecHash::operator()(const Vector3D& v) const {
    std::hash<double> h;
    return h(v.x * 73856093) ^ h(v.y * 19349663) ^ h(v.z * 83492791);
}
bool VecEq::operator()(const Vector3D& a, const Vector3D& b) const {
    return a == b;
}

Triangle::Triangle(int a, int b, int c) : v0(a), v1(b), v2(c) {}
Tetrahedron::Tetrahedron(int a, int b, int c, int d)
    : v0(a), v1(b), v2(c), v3(d) {}
Face::Face(int a, int b, int c) {
    v[0] = a; v[1] = b; v[2] = c;
    std::sort(v, v + 3);
}
bool Face::operator==(const Face& other) const {
    return v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2];
}
size_t FaceHash::operator()(const Face& f) const {
    return (f.v[0] * 73856093) ^ (f.v[1] * 19349663) ^ (f.v[2] * 83492791);
}

// --------------------------------------------------
// Static members
// --------------------------------------------------

// --------------------------------------------------
// STL loading
// --------------------------------------------------
void Mesh::loadSTL(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "❌ Error: Cannot open STL file " << filename << "\n";
        return;
    }

    char header[80];
    in.read(header, 80);

    uint32_t numTris = 0;
    in.read(reinterpret_cast<char*>(&numTris), sizeof(uint32_t));
    if (!in.good()) {
        std::cerr << "❌ Error reading STL header.\n";
        return;
    }

    points.clear();
    triangles.clear();
    vertexMap.clear();

    std::cout << "📂 Loading STL: " << filename << " (" << numTris << " triangles)\n";

    // Interpret STL coordinates as millimeters and keep as-is.
    for (uint32_t i = 0; i < numTris; ++i) {
        float nx, ny, nz;
        float vx[3], vy[3], vz[3];
        uint16_t attr;

        in.read(reinterpret_cast<char*>(&nx), 4);
        in.read(reinterpret_cast<char*>(&ny), 4);
        in.read(reinterpret_cast<char*>(&nz), 4);

        int vid[3];
        for (int j = 0; j < 3; ++j) {
            in.read(reinterpret_cast<char*>(&vx[j]), 4);
            in.read(reinterpret_cast<char*>(&vy[j]), 4);
            in.read(reinterpret_cast<char*>(&vz[j]), 4);
            vid[j] = addVertex(Vector3D((double)vx[j], (double)vy[j], (double)vz[j])); // mm
        }
        in.read(reinterpret_cast<char*>(&attr), 2);

        triangles.emplace_back(vid[0], vid[1], vid[2]);
    }

    in.close();

    computeBoundingBox();

    std::cout << "✅ STL loaded: " << points.size() << " unique vertices, "
        << triangles.size() << " triangles\n";
    std::cout << "?? Bounding Box (mm):\n";
    std::cout << "   Min: (" << minBounds.x << ", " << minBounds.y << ", " << minBounds.z << ")\n";
    std::cout << "   Max: (" << maxBounds.x << ", " << maxBounds.y << ", " << maxBounds.z << ")\n";
}


// --------------------------------------------------
// Add vertex with deduplication
// --------------------------------------------------
int Mesh::addVertex(const Vector3D& p) {
    auto key = toKey(p); // Rounds to 1e-9 precision as per your header
    auto it = vertexMap.find(key);
    if (it != vertexMap.end()) return it->second;
    int idx = (int)points.size();
    points.push_back(p);
    vertexMap[key] = idx;
    return idx;
}

//build BHV
int Mesh::buildBVH(int start, int end) {
    BVHNode node;
    AABB box; // auto-initialized to +/-1e18

    // Accumulate triangle bounds for [start,end)
    for (int i = start; i < end; ++i) {
        const int triId = triIndices[i];
        if (triId < 0 || triId >= (int)triangles.size()) continue;
        const Triangle& tri = triangles[triId];
        box.extend(points[tri.v0]); // mm
        box.extend(points[tri.v1]); // mm
        box.extend(points[tri.v2]); // mm
    }
    node.bounds = box;

    const int count = end - start;
    if (count <= 4) {
        node.start = start;
        node.range = count;
        const int idx = (int)bvhNodes.size();
        bvhNodes.push_back(node);
        return idx;
    }

    // Split by longest axis of box
    int axis = 0;
    Vector3D ext = Vector3D(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z);
    if (ext.y > ext.x) axis = 1;
    if (ext.z > ext[axis]) axis = 2;

    const double mid = 0.5 * (box.min[axis] + box.max[axis]);

    auto midIt = std::partition(triIndices.begin() + start,
        triIndices.begin() + end,
        [&](int triId) {
            const Triangle& t = triangles[triId];
            const double c = (points[t.v0][axis] + points[t.v1][axis] + points[t.v2][axis]) / 3.0;
            return c < mid;
        });

    int midIdx = (int)(midIt - triIndices.begin());
    if (midIdx <= start || midIdx >= end) midIdx = start + count / 2;

    // Reserve this node index, recurse
    const int curr = (int)bvhNodes.size();
    bvhNodes.push_back(node);

    int left = buildBVH(start, midIdx);
    int right = buildBVH(midIdx, end);

    bvhNodes[curr].left = left;
    bvhNodes[curr].right = right;
    bvhNodes[curr].start = 0;
    bvhNodes[curr].range = 0;

    return curr;
}


int Mesh::traverseBVH(int nodeIdx, const Vector3D& orig, const Vector3D& dir,
    const Vector3D& invDir, const std::array<int, 3>& dirIsNeg) {
    if (nodeIdx < 0 || nodeIdx >= (int)bvhNodes.size()) return 0;
    const BVHNode& node = bvhNodes[nodeIdx];
    if (!node.bounds.intersect(orig, invDir, dirIsNeg)) return 0;

    if (node.range > 0) {
        int hits = 0;
        int start = std::max(0, node.start);
        int end = std::min((int)triIndices.size(), node.start + node.range);
        for (int i = start; i < end; ++i) {
            const int triId = triIndices[i];
            if (triId < 0 || triId >= (int)triangles.size()) continue;
            double t;
            if (intersectTriangle(orig, dir, triangles[triId], t)) ++hits;
        }
        return hits;
    }
    const int lh = (node.left != -1) ? traverseBVH(node.left, orig, dir, invDir, dirIsNeg) : 0;
    const int rh = (node.right != -1) ? traverseBVH(node.right, orig, dir, invDir, dirIsNeg) : 0;
    return lh + rh;
}





// --------------------------------------------------
// Bounding box
// --------------------------------------------------
void Mesh::computeBoundingBox() {
    if (points.empty()) return;

    minBounds = Vector3D(1e18, 1e18, 1e18);
    maxBounds = Vector3D(-1e18, -1e18, -1e18);

    for (const auto& p : points) {
        minBounds.x = std::min(minBounds.x, p.x);
        minBounds.y = std::min(minBounds.y, p.y);
        minBounds.z = std::min(minBounds.z, p.z);
        maxBounds.x = std::max(maxBounds.x, p.x);
        maxBounds.y = std::max(maxBounds.y, p.y);
        maxBounds.z = std::max(maxBounds.z, p.z);
    }

    // Tiny padding only for numeric robustness; bounding box is NOT geometry
    const double span = std::max({ maxBounds.x - minBounds.x,
                                  maxBounds.y - minBounds.y,
                                  maxBounds.z - minBounds.z });
    const double margin = std::max(1e-9, 1e-6 * span);
    minBounds = minBounds - Vector3D(margin, margin, margin);
    maxBounds = maxBounds + Vector3D(margin, margin, margin);

    std::cout << "?? Bounding Box (mm):\n";
    std::cout << "   Min: (" << minBounds.x << ", " << minBounds.y << ", " << minBounds.z << ")\n";
    std::cout << "   Max: (" << maxBounds.x << ", " << maxBounds.y << ", " << maxBounds.z << ")\n";
}




bool Mesh::isWatertight() {
    std::unordered_map<std::pair<int, int>, int, pair_hash> edgeCount;

    auto makeEdge = [](int a, int b) {
        return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
    };

    for (const auto& tri : triangles) {
        std::pair<int, int> edges[3] = { makeEdge(tri.v0, tri.v1), makeEdge(tri.v1, tri.v2), makeEdge(tri.v2, tri.v0) };
        for (auto& e : edges)
            edgeCount[e]++;
    }

    std::vector<std::pair<int, int>> boundaryEdges;
    for (auto& [edge, count] : edgeCount)
        if (count != 2)
            boundaryEdges.push_back(edge);

    std::cout << "Mesh has " << boundaryEdges.size() << " boundary edges.\n";

    return boundaryEdges.empty();
}

bool Mesh::intersectTriangle(const Vector3D& orig, const Vector3D& dir,
    const Triangle& tri, double& t) {
    if (tri.v0 >= points.size() || tri.v1 >= points.size() || tri.v2 >= points.size()) {
        return false;
    }

    const Vector3D& v0 = points[tri.v0];
    const Vector3D& v1 = points[tri.v1];
    const Vector3D& v2 = points[tri.v2];

    const Vector3D e1 = v1 - v0;
    const Vector3D e2 = v2 - v0;

    // Scale epsilon to triangle size for mm geometry
    const double triSize = std::max({ e1.magnitude(), e2.magnitude(), (v2 - v1).magnitude() });
    const double EPS = std::max(1e-9, 1e-9 * triSize);

    const Vector3D h = dir.cross(e2);
    const double a = e1.dot(h);

    if (std::fabs(a) < EPS) return false; // ray parallel to triangle

    const double f = 1.0 / a;
    const Vector3D s = orig - v0;
    const double u = f * s.dot(h);

    if (u < 0.0 || u > 1.0) return false;

    const Vector3D q = s.cross(e1);
    const double v = f * dir.dot(q);

    if (v < 0.0 || u + v > 1.0) return false;

    t = f * e2.dot(q);
    return t > EPS; // only forward intersections
}


// --------------------------------------------------
// Grid sampling
// --------------------------------------------------
void Mesh::generateGridPoints(double dx) {
    std::cout << "🧊 Starting voxel grid generation...\n";
    std::cout << "[Debug] generateGridPoints dx = " << dx << " mm\n";

    insidePoints.clear();
    if (dx <= 0) {
        std::cerr << "❌ dx must be > 0\n";
        return;
    }

    // Use actual geometry bounds (not padded)
    double realMinX = 1e18, realMaxX = -1e18;
    double realMinY = 1e18, realMaxY = -1e18;
    double realMinZ = 1e18, realMaxZ = -1e18;

    for (const auto& p : points) {
        realMinX = std::min(realMinX, p.x);
        realMaxX = std::max(realMaxX, p.x);
        realMinY = std::min(realMinY, p.y);
        realMaxY = std::max(realMaxY, p.y);
        realMinZ = std::min(realMinZ, p.z);
        realMaxZ = std::max(realMaxZ, p.z);
    }

    const double Lx = realMaxX - realMinX;
    const double Ly = realMaxY - realMinY;
    const double Lz = realMaxZ - realMinZ;

    int nx = std::max(1, (int)std::ceil(Lx / dx));
    int ny = std::max(1, (int)std::ceil(Ly / dx));
    int nz = std::max(1, (int)std::ceil(Lz / dx));

    long long total = (long long)nx * (long long)ny * (long long)nz;

    std::cout << "📏 Grid: " << nx << " x " << ny << " x " << nz
        << " = " << total << " voxels\n";
    std::cout << "📏 Bounds: X[" << realMinX << "," << realMaxX << "] Y["
        << realMinY << "," << realMaxY << "] Z[" << realMinZ << "," << realMaxZ << "]\n";

    long long checked = 0;
    int insideCount = 0;

    for (int k = 0; k < nz; ++k) {
        const double z = realMinZ + (k + 0.5) * dx;
        for (int j = 0; j < ny; ++j) {
            const double y = realMinY + (j + 0.5) * dx;
            for (int i = 0; i < nx; ++i) {
                const double x = realMinX + (i + 0.5) * dx;
                const Vector3D p(x, y, z);

                if (pointInsideMesh(p, 1)) {
                    insidePoints.push_back(p);
                    insideCount++;
                }

                checked++;
                if (checked % 10000 == 0) {
                    double pct = (100.0 * checked) / (double)total;
                    std::cout << "   Progress: " << pct << "% (found " << insideCount << " interior)\n";
                }
            }
        }
    }

    std::cout << "✅ Voxel scan complete. Interior points: " << insidePoints.size() << "\n";

    if (insidePoints.empty()) {
        std::cerr << "⚠️ No interior points found!\n";
        std::cerr << "   Your cone should have interior volume. Check triangle winding.\n";
    }
}





void Mesh::logTriangleNormals() {
    std::cout << "\n=============================\n";
    std::cout << "?? Logging triangle normals and areas:\n";

    for (size_t i = 0; i < triangles.size(); ++i) {
        if (triangles[i].v0 >= points.size() ||
            triangles[i].v1 >= points.size() ||
            triangles[i].v2 >= points.size()) {
            std::cerr << "❌ Invalid triangle index at " << i << "\n";
            continue;
        }

        Vector3D v0 = points[triangles[i].v0];
        Vector3D v1 = points[triangles[i].v1];
        Vector3D v2 = points[triangles[i].v2];

        Vector3D normal = (v1 - v0).cross(v2 - v0);
        double area = normal.magnitude() * 0.5;

        if (area < 1e-15) {
            std::cout << "⚠️ Degenerate triangle at index " << i << "\n";
        }

        if (area > 0) {
            normal = normal.normalize();
        }

        std::cout << "Triangle " << i
            << " normal: (" << normal.x << ", " << normal.y << ", " << normal.z
            << "), area=" << area << "\n";
    }
    std::cout << "=============================\n";
}





// --------------------------------------------------
// Ray casting point-in-mesh
bool Mesh::pointInsideMesh(const Vector3D& p, int numRays) {
    // Test multiple ray directions to debug winding issues
    std::vector<Vector3D> testDirs = {
        Vector3D(1.0, 0.0, 0.0),   // +X
        Vector3D(-1.0, 0.0, 0.0),  // -X  
        Vector3D(0.0, 1.0, 0.0),   // +Y
        Vector3D(0.0, 0.0, 1.0)    // +Z
    };

    static int debugCount = 0;
    bool shouldDebug = (debugCount < 10);

    std::vector<int> hitCounts;

    // Cast rays in all test directions
    for (const auto& rayDir : testDirs) {
        Vector3D rayOrig = p + rayDir * (-1000.0); // start far away

        int hits = 0;
        for (const auto& tri : triangles) {
            double t;
            if (intersectTriangle(rayOrig, rayDir, tri, t)) {
                hits++;
            }
        }
        hitCounts.push_back(hits);
    }

    // Debug output for first few points
    if (shouldDebug) {
        std::cout << "[DEBUG] Point (" << p.x << "," << p.y << "," << p.z << ")\n";
        std::cout << "  +X hits=" << hitCounts[0] << " inside=" << ((hitCounts[0] & 1) == 0 && hitCounts[0] > 0 ? "YES" : "NO") << "\n";
        std::cout << "  -X hits=" << hitCounts[1] << " inside=" << ((hitCounts[1] & 1) == 0 && hitCounts[1] > 0 ? "YES" : "NO") << "\n";
        std::cout << "  +Y hits=" << hitCounts[2] << " inside=" << ((hitCounts[2] & 1) == 0 && hitCounts[2] > 0 ? "YES" : "NO") << "\n";
        std::cout << "  +Z hits=" << hitCounts[3] << " inside=" << ((hitCounts[3] & 1) == 0 && hitCounts[3] > 0 ? "YES" : "NO") << "\n";
        debugCount++;
    }

    // Count how many directions say "inside" using INVERTED PARITY
    // For inward-facing triangles: even hits (and > 0) = inside
    int insideVotes = 0;
    for (int hits : hitCounts) {
        if ((hits & 1) == 0 && hits > 0) {  // even and non-zero = inside
            insideVotes++;
        }
    }

    // If majority of rays say inside, trust that
    return insideVotes >= 2;
}




// ------------------ KDNode ------------------
KDNode::KDNode(int idx, const double* p) : index(idx), left(nullptr), right(nullptr) {
    point[0] = p[0]; point[1] = p[1]; point[2] = p[2];
}

// ------------------ KdTree ------------------
KdTree::KdTree() : root(nullptr) {}

KdTree::~KdTree() {
    clear(root);
}

void KdTree::clear(KDNode* node) {
    if (!node) return;
    clear(node->left);
    clear(node->right);
    delete node;
}

KDNode* KdTree::build(std::vector<std::pair<int, std::array<double, 3>>>& elems, int depth) {
    if (elems.empty()) return nullptr;

    int axis = depth % 3;
    size_t mid = elems.size() / 2;

    std::nth_element(elems.begin(), elems.begin() + mid, elems.end(),
        [axis](auto& a, auto& b) { return a.second[axis] < b.second[axis]; });

    KDNode* node = new KDNode(elems[mid].first, elems[mid].second.data());

    std::vector<std::pair<int, std::array<double, 3>>> left(elems.begin(), elems.begin() + mid);
    std::vector<std::pair<int, std::array<double, 3>>> right(elems.begin() + mid + 1, elems.end());

    node->left = build(left, depth + 1);
    node->right = build(right, depth + 1);

    return node;
}

void KdTree::buildTree(const std::vector<std::pair<int, std::array<double, 3>>>& elems) {
    auto copy = elems; // local copy because build modifies the vector
    root = build(copy);
}

double KdTree::distance(const double* a, const double* b) {
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) +
        (a[1] - b[1]) * (a[1] - b[1]) +
        (a[2] - b[2]) * (a[2] - b[2]));
}

void KdTree::radiusSearch(KDNode* node, const double* target, double radius,
    int depth, std::vector<int>& results) {
    if (!node) return;

    double dist = distance(node->point, target);
    if (dist <= radius) results.push_back(node->index);

    int axis = depth % 3;
    double diff = target[axis] - node->point[axis];

    if (diff <= radius) radiusSearch(node->left, target, radius, depth + 1, results);
    if (diff >= -radius) radiusSearch(node->right, target, radius, depth + 1, results);
}

std::vector<int> KdTree::query(const double* target, double radius) {
    std::vector<int> results;
    radiusSearch(root, target, radius, 0, results);
    return results;
}



void Mesh::filterGridPoints() {
    std::cout << "🧹 Filtering inside points...\n";
    const size_t interior = insidePoints.size();
    std::cout << "   Interior points: " << interior << "\n";

    if (interior == 0) {
        std::cerr << "⚠️ No interior points; skipping merge.\n";
        return;
    }

    // Optional dedup of inside points at 1e-9 mm tolerance
    auto keyFn = [](const Vector3D& v) {
        const double s = 1e9;
        return std::tuple<long long, long long, long long>(
            (long long)std::llround(v.x * s),
            (long long)std::llround(v.y * s),
            (long long)std::llround(v.z * s)
            );
    };
    std::unordered_set<std::tuple<long long, long long, long long>> seen;
    std::vector<Vector3D> unique;
    unique.reserve(insidePoints.size());
    for (const auto& v : insidePoints) {
        auto k = keyFn(v);
        if (seen.insert(k).second) unique.push_back(v);
    }
    insidePoints.swap(unique);

    std::cout << "✅ Final interior seed points: " << insidePoints.size() << "\n";
}







CircumSphere Mesh::computeCircumsphere(const Vector3D& A, const Vector3D& B,
    const Vector3D& C, const Vector3D& D)
{
    Vector3D AB = B - A;
    Vector3D AC = C - A;
    Vector3D AD = D - A;

    std::array<std::array<double, 3>, 3> M = { {
        {AB.x, AB.y, AB.z},
        {AC.x, AC.y, AC.z},
        {AD.x, AD.y, AD.z}
    } };

    std::array<double, 3> b = { {
        0.5 * AB.dot(AB),
        0.5 * AC.dot(AC),
        0.5 * AD.dot(AD)
    } };

    std::array<double, 3> sol; // temporary solution array
    const double EPS = 1e-12;

    // Gaussian elimination
    std::array<std::array<double, 3>, 3> tempM = M;
    std::array<double, 3> tempB = b;
    bool singular = false;

    for (int i = 0; i < 3; i++) {
        int maxRow = i;
        for (int k = i + 1; k < 3; k++)
            if (fabs(tempM[k][i]) > fabs(tempM[maxRow][i]))
                maxRow = k;
        std::swap(tempM[i], tempM[maxRow]);
        std::swap(tempB[i], tempB[maxRow]);

        if (fabs(tempM[i][i]) < EPS) {
            singular = true;
            break;
        }

        for (int k = i + 1; k < 3; k++) {
            double factor = tempM[k][i] / tempM[i][i];
            for (int j = i; j < 3; j++)
                tempM[k][j] -= factor * tempM[i][j];
            tempB[k] -= factor * tempB[i];
        }
    }

    if (singular) {
        return { Vector3D(0,0,0), -1.0 };
    }

    // Back-substitution
    for (int i = 2; i >= 0; i--) {
        double sum = tempB[i];
        for (int j = i + 1; j < 3; j++)
            sum -= tempM[i][j] * sol[j];
        sol[i] = sum / tempM[i][i];
    }

    Vector3D X(sol[0], sol[1], sol[2]);
    Vector3D center = A + X;
    double radius = (center - A).magnitude();

    return { center, radius };
}


bool Mesh::pointInCircumsphere(const Vector3D& p, const Tetrahedron& tet) {
    CircumSphere cs = computeCircumsphere(
        points[tet.v0], points[tet.v1], points[tet.v2], points[tet.v3]);

    double dist2 = (p - cs.center).dot(p - cs.center);
    double r2 = cs.radius * cs.radius;

    // Only consider strictly inside, with relative epsilon
    return dist2 < r2* (1.0 - 1e-12);
}

void Mesh::superTetra() {
    // Add 4 far-away auxiliary vertices and one super-tet (mm)
    const double sx = std::max(1e-9, maxBounds.x - minBounds.x);
    const double sy = std::max(1e-9, maxBounds.y - minBounds.y);
    const double sz = std::max(1e-9, maxBounds.z - minBounds.z);
    const double scale = 10.0;

    const Vector3D A(minBounds.x - scale * sx, minBounds.y - scale * sy, minBounds.z - scale * sz);
    const Vector3D B(maxBounds.x + scale * sx, minBounds.y - scale * sy, minBounds.z - scale * sz);
    const Vector3D C(minBounds.x - scale * sx, maxBounds.y + scale * sy, minBounds.z - scale * sz);
    const Vector3D D(minBounds.x - scale * sx, minBounds.y - scale * sy, maxBounds.z + scale * sz);

    addVertex(A);
    addVertex(B);
    addVertex(C);
    addVertex(D);

    const int n = (int)points.size();
    tets.emplace_back(n - 4, n - 3, n - 2, n - 1);
}


void Mesh::delaunayTetrahedralize() {
    std::cout << "⏳ Delaunay tetrahedralization (parallel-safe, mm-only)...\n";

    auto computeTetBoundingBox = [&]() -> std::pair<Vector3D, Vector3D> {
        Vector3D minB(1e18, 1e18, 1e18), maxB(-1e18, -1e18, -1e18);
        for (const auto& t : tets) {
            for (int vi : {t.v0, t.v1, t.v2, t.v3}) {
                if (vi >= 0 && vi < (int)points.size()) {
                    const Vector3D& p = points[vi];
                    minB.x = std::min(minB.x, p.x);
                    minB.y = std::min(minB.y, p.y);
                    minB.z = std::min(minB.z, p.z);
                    maxB.x = std::max(maxB.x, p.x);
                    maxB.y = std::max(maxB.y, p.y);
                    maxB.z = std::max(maxB.z, p.z);
                }
            }
        }
        return { minB, maxB };
    };

    const int auxStart = (int)points.size() - 4;
    const int auxEnd = auxStart + 4;

    // Unique STL surface vertices
    std::vector<int> surface;
    surface.reserve(triangles.size() * 3);
    for (const auto& tri : triangles) {
        surface.push_back(tri.v0);
        surface.push_back(tri.v1);
        surface.push_back(tri.v2);
    }
    std::sort(surface.begin(), surface.end());
    surface.erase(std::unique(surface.begin(), surface.end()), surface.end());

    auto insertPoint = [&](int pi, const Vector3D& p) {
        std::vector<int> badTets;
        badTets.reserve(256);

#pragma omp parallel
        {
            std::vector<int> local;
            local.reserve(256);
#pragma omp for schedule(static)
            for (int ti = 0; ti < (int)tets.size(); ++ti) {
                if (pointInCircumsphere(p, tets[ti])) local.push_back(ti);
            }
#pragma omp critical
            badTets.insert(badTets.end(), local.begin(), local.end());
        }

        if (badTets.empty()) return;

        std::unordered_map<Face, int, FaceHash> faceCount;
        faceCount.reserve(badTets.size() * 4);
        for (int idx : badTets) {
            if (idx < 0 || idx >= (int)tets.size()) continue;
            const Tetrahedron& t = tets[idx];
            Face f[4] = { Face(t.v0,t.v1,t.v2), Face(t.v0,t.v1,t.v3),
                          Face(t.v0,t.v2,t.v3), Face(t.v1,t.v2,t.v3) };
            for (auto& face : f) faceCount[face]++;
        }

        std::vector<Face> boundaryFaces;
        boundaryFaces.reserve(faceCount.size());
        for (const auto& kv : faceCount)
            if (kv.second == 1) boundaryFaces.push_back(kv.first);

        std::sort(badTets.begin(), badTets.end());
        badTets.erase(std::unique(badTets.begin(), badTets.end()), badTets.end());
        for (int i = (int)badTets.size() - 1; i >= 0; --i) {
            int idx = badTets[i];
            if (idx >= 0 && idx < (int)tets.size()) tets.erase(tets.begin() + idx);
        }

        for (const auto& f : boundaryFaces) {
            tets.emplace_back(f.v[0], f.v[1], f.v[2], pi);
        }
    };

    int insSurf = 0;
    for (int pi : surface) {
        if (pi < 0 || pi >= (int)points.size()) continue;
        if (pi >= auxStart && pi < auxEnd) continue; // skip auxiliary
        insertPoint(pi, points[pi]);
        if (++insSurf % 50 == 0 || insSurf == (int)surface.size()) {
            std::cout << "[Surface Insert] " << insSurf << " / "
                << surface.size() << " | Tets: " << tets.size() << "\n";
        }
    }

    const int batchSize = 512;
    const int totalPts = (int)insidePoints.size();
    const int nBatches = (totalPts + batchSize - 1) / batchSize;

    auto baselineBB = computeTetBoundingBox();
    int baseCount = (int)tets.size();
    auto safeBackup = tets;

    int insInterior = 0;
    for (int b = 0; b < nBatches; ++b) {
        const int s = b * batchSize;
        const int e = std::min(s + batchSize, totalPts);

        for (int i = s; i < e; ++i) {
            const Vector3D p = insidePoints[i];
            const int pi = addVertex(p);
            insertPoint(pi, p);
            ++insInterior;
        }

        auto nowBB = computeTetBoundingBox();
        const double dx = (nowBB.second.x - baselineBB.second.x) / std::max(1e-12, (baselineBB.second.x - baselineBB.first.x));
        const double dy = (nowBB.second.y - baselineBB.second.y) / std::max(1e-12, (baselineBB.second.y - baselineBB.first.y));
        const double dz = (nowBB.second.z - baselineBB.second.z) / std::max(1e-12, (baselineBB.second.z - baselineBB.first.z));
        const int nowCount = (int)tets.size();

        if (dx > 0.2 || dy > 0.2 || dz > 0.2 || nowCount > baseCount * 6) {
            std::cout << "⚠️ Distortion at batch " << b << " — rolling back.\n";
            tets = safeBackup;
            break;
        }
        else {
            safeBackup = tets;
            baselineBB = nowBB;
            baseCount = nowCount;
            std::cout << "[Batch " << b << "] Inserted interior: " << insInterior
                << " | Tets: " << nowCount << "\n";
        }
    }

    // Remove any tet containing auxiliary vertices
    tets.erase(std::remove_if(tets.begin(), tets.end(),
        [&](const Tetrahedron& t) {
            return (t.v0 >= auxStart && t.v0 < auxEnd) ||
                (t.v1 >= auxStart && t.v1 < auxEnd) ||
                (t.v2 >= auxStart && t.v2 < auxEnd) ||
                (t.v3 >= auxStart && t.v3 < auxEnd);
        }), tets.end());

    std::cout << "✅ Delaunay tetrahedralization complete. " << tets.size() << " tets.\n";
}





void Mesh::patchifySurface(double normalThresholdDeg) {
    double cosThreshold = std::cos(normalThresholdDeg * M_PI / 180.0);
    std::vector<bool> visited(triangles.size(), false);
    patches.clear();

    // --- Build adjacency list for triangles (shared edge → neighbors)
    std::unordered_map<Face, std::vector<int>, FaceHash> edgeToTris;
    for (int tid = 0; tid < (int)triangles.size(); ++tid) {
        const Triangle& t = triangles[tid];
        int v[3] = { t.v0, t.v1, t.v2 };
        for (int i = 0; i < 3; i++) {
            Face e(v[i], v[(i + 1) % 3], -1); // edge (two verts + -1 dummy)
            edgeToTris[e].push_back(tid);
        }
    }

    auto triNormal = [&](const Triangle& tri) {
        Vector3D v0 = points[tri.v0];
        Vector3D v1 = points[tri.v1];
        Vector3D v2 = points[tri.v2];
        return (v1 - v0).cross(v2 - v0).normalize();
    };

    for (int i = 0; i < (int)triangles.size(); ++i) {
        if (visited[i]) continue;

        Patch patch;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;

        Vector3D seedNormal = triNormal(triangles[i]);

        while (!q.empty()) {
            int tid = q.front(); q.pop();
            patch.triangles.push_back(tid);

            Vector3D nTri = triNormal(triangles[tid]);
            int v[3] = { triangles[tid].v0, triangles[tid].v1, triangles[tid].v2 };
            for (int e = 0; e < 3; e++) {
                Face edge(v[e], v[(e + 1) % 3], -1);
                auto it = edgeToTris.find(edge);
                if (it == edgeToTris.end()) continue;

                for (int nbr : it->second) {
                    if (!visited[nbr]) {
                        Vector3D nNbr = triNormal(triangles[nbr]);
                        if (nTri.dot(nNbr) > cosThreshold) {
                            visited[nbr] = true;
                            q.push(nbr);
                        }
                    }
                }
            }
        }

        std::set<int> verts;
        for (int tid : patch.triangles) {
            verts.insert(triangles[tid].v0);
            verts.insert(triangles[tid].v1);
            verts.insert(triangles[tid].v2);
        }

        Vector3D centroid(0, 0, 0);
        for (int vid : verts) centroid = centroid + points[vid];
        centroid = centroid * (1.0 / verts.size());
        patch.centroid = centroid;

        Vector3D avgNormal(0, 0, 0);
        for (int tid : patch.triangles) {
            avgNormal = avgNormal + triNormal(triangles[tid]);
        }
        patch.normal = avgNormal.normalize();

        double meanX = 0, meanY = 0, meanZ = 0;
        for (int vid : verts) {
            meanX += points[vid].x;
            meanY += points[vid].y;
            meanZ += points[vid].z;
        }
        meanX /= verts.size();
        meanY /= verts.size();
        meanZ /= verts.size();

        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) patch.cov[r][c] = 0.0;
        for (int vid : verts) {
            double dx = points[vid].x - meanX;
            double dy = points[vid].y - meanY;
            double dz = points[vid].z - meanZ;
            patch.cov[0][0] += dx * dx; patch.cov[0][1] += dx * dy; patch.cov[0][2] += dx * dz;
            patch.cov[1][0] += dy * dx; patch.cov[1][1] += dy * dy; patch.cov[1][2] += dy * dz;
            patch.cov[2][0] += dz * dx; patch.cov[2][1] += dz * dy; patch.cov[2][2] += dz * dz;
        }
        double n = (double)verts.size() - 1.0;
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) patch.cov[r][c] /= n;

        Vector3D normal = patch.normal;
        Vector3D uAxis;
        if (std::fabs(normal.x) > std::fabs(normal.y))
            uAxis = Vector3D(-normal.z, 0, normal.x).normalize();
        else
            uAxis = Vector3D(0, normal.z, -normal.y).normalize();
        Vector3D vAxis = normal.cross(uAxis).normalize();
        patch.uAxis = uAxis;
        patch.vAxis = vAxis;

        patches.push_back(patch);
    }

    std::cout << "✅ Patchified surface into " << patches.size() << " patches.\n";
    for (size_t i = 0; i < patches.size(); ++i) {
        std::cout << "Patch " << i << " triangle count: " << patches[i].triangles.size() << std::endl;
    }
}






void Mesh::remeshPatch(Patch& patch) {
    std::vector<int> patchVerts;
    std::set<int> vertSet;

    // Collect unique vertices in the patch
    for (auto tid : patch.triangles) {
        vertSet.insert(triangles[tid].v0);
        vertSet.insert(triangles[tid].v1);
        vertSet.insert(triangles[tid].v2);
    }

    for (auto vid : vertSet)
        patchVerts.push_back(vid);

    // Project vertices to 2D plane
    struct Point2D { double x, y; Point2D(double xx, double yy) : x(xx), y(yy) {} };
    struct Tri2D { int a, b, c; Tri2D(int aa, int bb, int cc) : a(aa), b(bb), c(cc) {} };
    std::vector<Point2D> points2D;

    for (auto vid : patchVerts) {
        double u, w;
        projectTo2D(points[vid], patch, u, w);
        points2D.emplace_back(u, w);
    }

    // Simple fan triangulation in 2D
    std::vector<Tri2D> tris2D;
    for (size_t i = 1; i + 1 < points2D.size(); i++)
        tris2D.emplace_back(0, i, i + 1);

    // Convert 2D triangles back to 3D and add to mesh
    for (auto& t2 : tris2D) {
        int a3D = addVertex(projectTo3D(points2D[t2.a].x, points2D[t2.a].y, patch));
        int b3D = addVertex(projectTo3D(points2D[t2.b].x, points2D[t2.b].y, patch));
        int c3D = addVertex(projectTo3D(points2D[t2.c].x, points2D[t2.c].y, patch));
        triangles.emplace_back(a3D, b3D, c3D);
    }
}


void Mesh::remeshSurfacePatches() {
    for (auto& p : patches)
        remeshPatch(p);

    std::cout << "Surface remeshing done. Triangles count: " << triangles.size() << "\n";
    if (triangles.empty()) {
        std::cerr << "No triangles generated after remeshing. Check patch remeshing logic." << std::endl;
    }
}


//vtk
void Mesh::writeVTK(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "❌ Cannot open file for VTK output: " << filename << "\n";
        return;
    }

    if (points.empty() && insidePoints.empty()) {
        std::cerr << "⚠️ Warning: No points to write in VTK file.\n";
    }

    // Combine surface points and interior points for VTK
    std::vector<Vector3D> allPoints = points;
    allPoints.insert(allPoints.end(), insidePoints.begin(), insidePoints.end());

    out << "# vtk DataFile Version 3.0\nMesh generated by CFD Mesher\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    out << "POINTS " << allPoints.size() << " double\n";
    for (auto& p : allPoints)
        out << p.x << " " << p.y << " " << p.z << "\n";

    if (!tets.empty()) {
        out << "CELLS " << tets.size() << " " << tets.size() * 5 << "\n";
        for (auto& t : tets)
            out << "4 " << t.v0 << " " << t.v1 << " " << t.v2 << " " << t.v3 << "\n";

        out << "CELL_TYPES " << tets.size() << "\n";
        for (size_t i = 0; i < tets.size(); i++)
            out << "10\n";
    }

    out.close();

    std::cout << "✅ VTK file written: " << filename << "\n";
    std::cout << "Surface points: " << points.size()
        << ", Interior points: " << insidePoints.size()
        << ", Tetrahedra: " << tets.size() << "\n";
}



void Mesh::repairDeduplicate() {
    std::vector<Triangle> cleanTriangles;
    for (auto& tri : triangles) {
        Vector3D v0 = points[tri.v0];
        Vector3D v1 = points[tri.v1];
        Vector3D v2 = points[tri.v2];
        Vector3D edge1 = v1 - v0;
        Vector3D edge2 = v2 - v0;
        Vector3D normal = edge1.cross(edge2);
        if (normal.magnitude() > 1e-12) {
            cleanTriangles.push_back(tri);
        }
    }
    triangles = std::move(cleanTriangles);
    std::cout << "Removed degenerate triangles, now " << triangles.size() << " triangles left.\n";
}


void Mesh::fixBoundaryOrientation() {
    Vector3D avgNormal(0, 0, 0);
    for (auto& tri : triangles) {
        Vector3D v0 = points[tri.v0], v1 = points[tri.v1], v2 = points[tri.v2];
        Vector3D normal = (v1 - v0).cross(v2 - v0).normalize();
        avgNormal = avgNormal + normal;
    }
    avgNormal = avgNormal.normalize();
    Vector3D center = (minBounds + maxBounds) * 0.5;

    for (auto& tri : triangles) {
        Vector3D v0 = points[tri.v0];
        Vector3D dir = center - v0;
        Vector3D normal = (points[tri.v1] - v0).cross(points[tri.v2] - v0);
        if (normal.dot(dir) > 0)
            std::swap(tri.v1, tri.v2); // flip!
    }
    std::cout << "Fixed triangle orientation to outward normals.\n";
}





// Entry point
int GenerateMeshFromConsoleInput() {
    std::string filename;
    double dx_mm;

    std::cout << "Enter STL filename: ";
    std::getline(std::cin, filename);

    std::cout << "Enter grid spacing in mm: ";
    std::cin >> dx_mm;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Keep millimeters throughout (no /1000 anywhere)
    const double dx = dx_mm;
    std::cout << "[Debug] Using dx = " << dx << " mm\n";

    Mesh mesh;
    try {
        mesh.loadSTL(filename);            // geometry stays in mm
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    // Add this test right after STL loading
    std::vector<Vector3D> testPoints = {
    Vector3D(28.0, 28.0, 50.0),   // near cone tip
    Vector3D(20.0, 20.0, 200.0),  // middle of cone  
    Vector3D(15.0, 15.0, 400.0),  // cylinder section
    Vector3D(25.0, 25.0, 600.0)   // flare section
    };

    std::cout << "=== TESTING KNOWN INTERIOR POINTS ===\n";
    for (const auto& pt : testPoints) {
        bool isInside = mesh.pointInsideMesh(pt, 1);
        std::cout << "Point (" << pt.x << "," << pt.y << "," << pt.z << ") inside: " << (isInside ? "YES" : "NO") << "\n";
    }
    std::cout << "=====================================\n";


    if (!mesh.isWatertight()) {
        std::cerr << "⚠️ Mesh not watertight: holes or open boundaries detected!\n";
    }
    else {
        std::cout << "✅ Mesh is watertight.\n";
    }

    mesh.logTriangleNormals();
    mesh.repairDeduplicate();
    mesh.computeBoundingBox();
    //mesh.fixBoundaryOrientation();

    // Surface processing (still mm)
    mesh.patchifySurface(10.0);
    mesh.remeshSurfacePatches();

    // Build BVH over STL triangles (NOT bounding box)
    mesh.triIndices.resize(mesh.triangles.size());
    for (int i = 0; i < (int)mesh.triangles.size(); ++i) mesh.triIndices[i] = i;
    mesh.rootBVHNode = mesh.buildBVH(0, (int)mesh.triangles.size());
    if (mesh.bvhNodes.empty()) {
        std::cerr << "❌ BVH build failed\n";
        return 1;
    }
    std::cout << "BVH built. Root=" << mesh.rootBVHNode
        << ", nodes=" << mesh.bvhNodes.size() << "\n";

    // Interior sampling (mm); writes voxel centers to insidePoints only
    mesh.generateGridPoints(dx);
    mesh.filterGridPoints();

    // Volume mesh (auxiliary super-tet removed at end inside Delaunay)
    mesh.superTetra();
    mesh.delaunayTetrahedralize();

    mesh.writeVTK("mesh.vtk");
    std::cout << "Mesh generated and saved to mesh.vtk\n";
    return 0;
}


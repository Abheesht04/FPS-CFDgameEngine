#include "postproc.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include<cmath>
#include <algorithm>

void postproc::loadFromVTK(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    points_.clear();
    cells_.clear();

    std::string line;
    while (std::getline(file, line))
    {
        // Read points
        if (line.rfind("POINTS", 0) == 0)
        {
            std::istringstream iss(line);
            std::string keyword;
            std::size_t npoints;
            std::string dtype;
            iss >> keyword >> npoints >> dtype;

            points_.reserve(npoints);

            for (std::size_t i = 0; i < npoints; ++i) {
                double x, y, z;
                if (!(file >> x >> y >> z)) {
                    throw std::runtime_error("Error reading point at index " + std::to_string(i));
                }
                points_.push_back({ x, y, z });
            }
            std::getline(file, line); // consume trailing newline
        }

        // Read cells
        if (line.rfind("CELLS", 0) == 0)
        {
            std::istringstream iss(line);
            std::string keyword;
            std::size_t ncells, size;
            iss >> keyword >> ncells >> size;

            cells_.reserve(ncells);

            for (std::size_t i = 0; i < ncells; ++i) {
                int nverts;
                if (!(file >> nverts)) {
                    throw std::runtime_error("Error reading number of vertices for cell " + std::to_string(i));
                }
                Cell c;
                c.nodeIds.resize(nverts);
                for (int j = 0; j < nverts; ++j) {
                    if (!(file >> c.nodeIds[j])) {
                        throw std::runtime_error("Error reading node index for cell " + std::to_string(i));
                    }
                }
                cells_.push_back(std::move(c));
            }
            std::getline(file, line);
        }

        // Read cell types
        if (line.rfind("CELL_TYPES", 0) == 0)
        {
            std::istringstream iss(line);
            std::string keyword;
            std::size_t ncells;
            iss >> keyword >> ncells;

            if (ncells != cells_.size()) {
                throw std::runtime_error("CELL_TYPES count does not match number of cells");
            }

            for (std::size_t i = 0; i < ncells; ++i) {
                int type;
                if (!(file >> type)) {
                    throw std::runtime_error("Error reading cell type at index " + std::to_string(i));
                }
                cells_[i].type = type;
            }
            std::getline(file, line);
        }
    }
}


BoundingBox computeBoundingBox(const std::vector<Point3D>& pts) {
    if (pts.empty()) throw std::runtime_error("No points");

    BoundingBox box{ pts[0], pts[0] };
    for (auto& p : pts) {
        box.min.x = std::min(box.min.x, p.x);
        box.min.y = std::min(box.min.y, p.y);
        box.min.z = std::min(box.min.z, p.z);
        box.max.x = std::max(box.max.x, p.x);
        box.max.y = std::max(box.max.y, p.y);
        box.max.z = std::max(box.max.z, p.z);
    }
    return box;
}

Point3D computeCentroid(const std::vector<Point3D>& pts) {
    Point3D c{ 0,0,0 };
    for (auto& p : pts) {
        c.x += p.x;
        c.y += p.y;
        c.z += p.z;
    }
    c.x /= pts.size();
    c.y /= pts.size();
    c.z /= pts.size();
    return c;
}

void translateMesh(std::vector<Point3D>& pts, const Point3D& shift) {
    for (auto& p : pts) {
        p.x += shift.x;
        p.y += shift.y;
        p.z += shift.z;
    }
}

void checkOrientation(const BoundingBox& box) {
    double dx = box.max.x - box.min.x;
    double dy = box.max.y - box.min.y;
    double dz = box.max.z - box.min.z;

    std::cout << "Extent X=" << dx << " Y=" << dy << " Z=" << dz << "\n";

    if (dx < 1e-6) std::cout << "Mesh lies in YZ plane (X˜0)\n";
    if (dy < 1e-6) std::cout << "Mesh lies in XZ plane (Y˜0)\n";
    if (dz < 1e-6) std::cout << "Mesh lies in XY plane (Z˜0)\n";
}


bool FaceKey::operator==(const FaceKey& other) const {
    return nodes == other.nodes;
}

std::size_t FaceKeyHash::operator()(const FaceKey& key) const {
    std::size_t h = 0;
    for (auto n : key.nodes) {
        h ^= std::hash<int>{}(n)+0x9e37779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

Point3D computeFaceCenter(const std::vector<int>& nodeIds, const std::vector<Point3D>& points) {
    Point3D c{ 0,0,0 };
    for (auto nid : nodeIds) {
        c.x += points[nid].x;
        c.y += points[nid].y;
        c.z += points[nid].z;
    }
    double inv = 1.0 / nodeIds.size();
    c.x *= inv; c.y *= inv; c.z *= inv;
    return c;
}

Point3D computeFaceNormalAndArea(const std::vector<int>& nodeIds, const std::vector<Point3D>& points, double& area) {
    Point3D normal{ 0,0,0 };
    if (nodeIds.size() == 3) {
        // triangle
        const auto& A = points[nodeIds[0]];
        const auto& B = points[nodeIds[1]];
        const auto& C = points[nodeIds[2]];
        Point3D u{ B.x - A.x, B.y - A.y, B.z - A.z };
        Point3D v{ C.x - A.x, C.y - A.y, C.z - A.z };
        normal.x = u.y * v.z - u.z * v.y;
        normal.y = u.z * v.x - u.x * v.z;
        normal.z = u.x * v.y - u.y * v.x;
        area = 0.5 * std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        // normalize
        double len = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (len > 1e-12) { normal.x /= len; normal.y /= len; normal.z /= len; }
    }
    else if (nodeIds.size() == 4) {
        // quad: average of two triangles
        const auto& A = points[nodeIds[0]];
        const auto& B = points[nodeIds[1]];
        const auto& C = points[nodeIds[2]];
        const auto& D = points[nodeIds[3]];
        Point3D n1{ (B.y - A.y) * (C.z - A.z) - (B.z - A.z) * (C.y - A.y),
                    (B.z - A.z) * (C.x - A.x) - (B.x - A.x) * (C.z - A.z),
                    (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x) };
        Point3D n2{ (D.y - A.y) * (C.z - A.z) - (D.z - A.z) * (C.y - A.y),
                    (D.z - A.z) * (C.x - A.x) - (D.x - A.x) * (C.z - A.z),
                    (D.x - A.x) * (C.y - A.y) - (D.y - A.y) * (C.x - A.x) };
        normal.x = n1.x + n2.x;
        normal.y = n1.y + n2.y;
        normal.z = n1.z + n2.z;
        area = 0.5 * (std::sqrt(n1.x * n1.x + n1.y * n1.y + n1.z * n1.z) +
            std::sqrt(n2.x * n2.x + n2.y * n2.y + n2.z * n2.z));
        double len = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (len > 1e-12) { normal.x /= len; normal.y /= len; normal.z /= len; }
    }
    else {
        throw std::runtime_error("Unsupported face with nodes != 3 or 4");
    }
    return normal;
}

// -------------------- Build Face Connectivity --------------------
void postproc::buildFaceConnectivity() {
    // clear previous
    faces_.clear();
    faceMap_.clear();
    nodeToFace_.clear();
    nodeToFace_.resize(points_.size());

    for (std::size_t ci = 0; ci < cells_.size(); ++ci) {
        const Cell& cell = cells_[ci];
        auto cellFaces = getCellFaces(cell);

        for (auto& fNodes : cellFaces) {
            auto keyNodes = fNodes;
            std::sort(keyNodes.begin(), keyNodes.end());
            FaceKey fk{ keyNodes };

            auto it = faceMap_.find(fk);
            if (it == faceMap_.end()) {
                Face f;
                f.nodeIds = fNodes;
                f.leftCell = static_cast<int>(ci);

                // compute center, normal, area
                f.center = computeFaceCenter(f.nodeIds, points_);
                f.normal = computeFaceNormalAndArea(f.nodeIds, points_, f.area);

                int fid = static_cast<int>(faces_.size());
                faces_.push_back(f);
                faceMap_[fk] = fid;

                for (auto n : fNodes) nodeToFace_[n].push_back(fid);
            }
            else {
                int fid = it->second;
                faces_[fid].rightCell = static_cast<int>(ci);
            }
        }
    }
}


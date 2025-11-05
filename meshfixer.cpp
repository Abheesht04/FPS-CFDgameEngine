// meshfixer.cpp
#include "meshfixer.h"
#include "mesh.h"
#include <algorithm>
#include <map>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdlib>

// ------------------------ Constructor ------------------------
MeshFixer::MeshFixer(const std::vector<Node>& nodes, const std::vector<Triangle>& faces)
    : nodes_(nodes), faces_(faces)
{
}

// ------------------------ Utilities ------------------------
double MeshFixer::tri_area(const Node& a, const Node& b, const Node& c) {
    double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
    double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
    double cx = uy * vz - uz * vy;
    double cy = uz * vx - ux * vz;
    double cz = ux * vy - uy * vx;
    return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

void MeshFixer::bbox_nodes(const std::vector<Node>& V, Node& mn, Node& mx) {
    if (V.empty()) return;
    mn = mx = V[0];
    for (const auto& n : V) {
        if (n.x < mn.x) mn.x = n.x;
        if (n.y < mn.y) mn.y = n.y;
        if (n.z < mn.z) mn.z = n.z;
        if (n.x > mx.x) mx.x = n.x;
        if (n.y > mx.y) mx.y = n.y;
        if (n.z > mx.z) mx.z = n.z;
    }
}

// ------------------------ Repair Methods ------------------------
void MeshFixer::weld_vertices(double tol) {
    std::vector<Node> new_nodes;
    std::map<int, int> mapping;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        bool found = false;
        for (size_t j = 0; j < new_nodes.size(); ++j) {
            double dx = nodes_[i].x - new_nodes[j].x;
            double dy = nodes_[i].y - new_nodes[j].y;
            double dz = nodes_[i].z - new_nodes[j].z;
            if (dx * dx + dy * dy + dz * dz < tol * tol) {
                mapping[i] = (int)j;
                found = true;
                break;
            }
        }
        if (!found) {
            mapping[i] = (int)new_nodes.size();
            new_nodes.push_back(nodes_[i]);
        }
    }
    for (auto& tri : faces_) {
        tri.v0 = mapping[tri.v0];
        tri.v1 = mapping[tri.v1];
        tri.v2 = mapping[tri.v2];
    }
    nodes_ = new_nodes;
}

void MeshFixer::remove_degenerate(double area_tol) {
    std::vector<Triangle> new_faces;
    for (auto& tri : faces_) {
        double a = tri_area(nodes_[tri.v0], nodes_[tri.v1], nodes_[tri.v2]);
        if (a > area_tol) new_faces.push_back(tri);
    }
    faces_ = new_faces;
}

void MeshFixer::orient_consistently() {
    for (auto& tri : faces_) {
        Node& a = nodes_[tri.v0];
        Node& b = nodes_[tri.v1];
        Node& c = nodes_[tri.v2];
        double cx = (b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y);
        double cy = (b.z - a.z) * (c.x - a.x) - (b.x - a.x) * (c.z - a.z);
        double cz = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (cx + cy + cz < 0) std::swap(tri.v1, tri.v2);
    }
}

bool MeshFixer::is_watertight(size_t& border_edges) const {
    std::map<std::pair<int, int>, int> edge_count;
    for (auto& tri : faces_) {
        int v[3] = { tri.v0, tri.v1, tri.v2 };
        for (int i = 0; i < 3; ++i) {
            int a = std::min(v[i], v[(i + 1) % 3]);
            int b = std::max(v[i], v[(i + 1) % 3]);
            edge_count[{a, b}]++;
        }
    }
    border_edges = 0;
    for (auto& kv : edge_count) {
        if (kv.second != 2) border_edges++;
    }
    return border_edges == 0;
}

void MeshFixer::repair_mesh() {
    weld_vertices(1e-6);
    remove_degenerate(1e-18);
    orient_consistently();
}

void MeshFixer::get_repaired(std::vector<Node>& nodes_out, std::vector<Triangle>& faces_out) const {
    nodes_out = nodes_;
    faces_out = faces_;
}

// ------------------------ STL Loader ------------------------
bool load_stl(const std::string& filename,
    std::vector<Node>& nodes,
    std::vector<Triangle>& faces)
{
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) return false;

    char header[80]; fin.read(header, 80);
    uint32_t tri_count; fin.read(reinterpret_cast<char*>(&tri_count), 4);

    if (!fin) { // fallback ASCII
        fin.close();
        std::ifstream fin2(filename);
        if (!fin2) return false;

        nodes.clear(); faces.clear();
        std::map<std::tuple<double, double, double>, int> node_map;
        std::string line;
        while (std::getline(fin2, line)) {
            std::istringstream iss(line);
            std::string token; iss >> token;
            if (token == "vertex") {
                double x, y, z; iss >> x >> y >> z;
                auto key = std::make_tuple(x, y, z);
                if (node_map.count(key) == 0) {
                    node_map[key] = (int)nodes.size();
                    nodes.push_back({ x,y,z });
                }
            }
            else if (token == "endloop") {
                int n = (int)nodes.size();
                if (n >= 3) faces.push_back({ n - 3,n - 2,n - 1 });
            }
        }
        return true;
    }

    nodes.clear(); faces.clear();
    fin.seekg(84, std::ios::beg);
    std::map<std::tuple<float, float, float>, int> node_map;
    for (uint32_t i = 0; i < tri_count; i++) {
        float normal[3], v[3][3];
        fin.read(reinterpret_cast<char*>(normal), 12);
        for (int j = 0; j < 3; j++) fin.read(reinterpret_cast<char*>(v[j]), 12);
        fin.ignore(2);
        int idx[3];
        for (int j = 0; j < 3; j++) {
            auto key = std::make_tuple(v[j][0], v[j][1], v[j][2]);
            if (node_map.count(key) == 0) {
                node_map[key] = (int)nodes.size();
                nodes.push_back({ v[j][0],v[j][1],v[j][2] });
            }
            idx[j] = node_map[key];
        }
        faces.push_back({ idx[0],idx[1],idx[2] });
    }
    return true;
}



bool MeshFixer::export_stl(const std::string& filename) const {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "[Error] Cannot write STL file: " << filename << "\n";
        return false;
    }

    fout << "solid repaired_mesh\n";
    for (const auto& tri : faces_) {
        const Node& a = nodes_[tri.v0];
        const Node& b = nodes_[tri.v1];
        const Node& c = nodes_[tri.v2];

        // Compute normal
        double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        double nx = uy * vz - uz * vy;
        double ny = uz * vx - ux * vz;
        double nz = ux * vy - uy * vx;
        double len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-12) { nx /= len; ny /= len; nz /= len; }
        else { nx = ny = nz = 0; }

        fout << "  facet normal " << nx << " " << ny << " " << nz << "\n";
        fout << "    outer loop\n";
        fout << "      vertex " << a.x << " " << a.y << " " << a.z << "\n";
        fout << "      vertex " << b.x << " " << b.y << " " << b.z << "\n";
        fout << "      vertex " << c.x << " " << c.y << " " << c.z << "\n";
        fout << "    endloop\n";
        fout << "  endfacet\n";
    }
    fout << "endsolid repaired_mesh\n";
    fout.close();
    std::cout << "[Info] Exported repaired mesh to " << filename << "\n";
    return true;
}


// ------------------------ Main ------------------------
int main() {
    std::cout << "=== QuickMesh with MeshFixer ===\n";
    std::string filename;
    std::cout << "Enter STL filename: ";
    std::getline(std::cin, filename);

    // Load STL
    std::vector<Node> nodes;
    std::vector<Triangle> triangles;
    if (!load_stl(filename, nodes, triangles)) {
        std::cerr << "[Error] Failed to load STL.\n";
        return 1;
    }
    std::cout << "[Info] STL loaded: " << nodes.size() << " vertices, " << triangles.size() << " triangles\n";

    // Repair mesh
    MeshFixer fixer(nodes, triangles);
    fixer.repair_mesh();
    fixer.get_repaired(nodes, triangles);

    size_t border_edges;
    bool watertight = fixer.is_watertight(border_edges);
    std::cout << "[Info] Mesh repaired. Border edges: " << border_edges
        << ", Watertight: " << (watertight ? "Yes" : "No") << "\n";

    // Export repaired STL
    std::string repaired_stl = "new_mesh_fixed.stl";
    fixer.export_stl(repaired_stl);
    std::cout << "[Info] Exported repaired mesh to " << repaired_stl << "\n";

    // Mesh quality input
    std::cout << "Choose mesh quality:\n1. Coarse\n2. Medium\n3. Fine\nEnter option [1-3]: ";
    int opt;
    std::cin >> opt;

    Mesher::MeshQuality mq = Mesher::MeshQuality::Medium;
    switch (opt) {
    case 1: mq = Mesher::MeshQuality::Coarse; break;
    case 2: mq = Mesher::MeshQuality::Medium; break;
    case 3: mq = Mesher::MeshQuality::Fine; break;
    default: std::cout << "[Info] Defaulting to Medium.\n"; break;
    }

    // Characteristic length input (in mm)
    double char_len;
    std::cout << "Enter characteristic length for tetrahedra (mm): ";
    std::cin >> char_len;

    // Generate Tet mesh
    std::string vtk_file = "tet_mesh";
    Mesher mesher;
    if (mesher.mesh_stl_with_tetgen(repaired_stl, vtk_file, char_len, mq)) {
        std::cout << "[Info] Tet mesh generation successful!\n";
    }
    else {
        std::cerr << "[Error] Tet mesh generation failed!\n";
    }

    return 0;
}


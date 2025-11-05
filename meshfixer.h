#pragma once
#include <vector>
#include <string>

// -------------------- Node and Triangle --------------------
struct Node {
    double x, y, z;
};

struct Triangle {
    int v0, v1, v2;
};

// -------------------- MeshFixer --------------------
class MeshFixer {
public:
    // Constructor
    MeshFixer(const std::vector<Node>& nodes, const std::vector<Triangle>& faces);

    // Utilities
    static double tri_area(const Node& a, const Node& b, const Node& c);
    static void bbox_nodes(const std::vector<Node>& V, Node& mn, Node& mx);

    // Repair methods
    void weld_vertices(double tol);
    void remove_degenerate(double area_tol);
    void orient_consistently();
    void repair_mesh();

    // Check watertightness
    bool is_watertight(size_t& border_edges) const;


    // Get repaired mesh
    void get_repaired(std::vector<Node>& nodes_out, std::vector<Triangle>& faces_out) const;

    // Export to ASCII STL
    bool export_stl(const std::string& filename) const;

private:
    std::vector<Node> nodes_;
    std::vector<Triangle> faces_;
};

#pragma once
#include "Mesher.h"
#include <d3d11.h>
#include <DirectXMath.h>
#include <vector>

class Viewer
{
private:
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;

    ID3D11VertexShader* vertexShader = nullptr;
    ID3D11PixelShader* pixelShader = nullptr;
    ID3D11InputLayout* inputLayout = nullptr;

    ID3D11Buffer* vertexBuffer = nullptr;
    ID3D11Buffer* indexBuffer = nullptr;
    UINT indexCount = 0;

    // Constant buffer for world-view-projection matrix
    ID3D11Buffer* constantBuffer = nullptr;

    struct ConstantBuffer
    {
        DirectX::XMMATRIX worldViewProj;
    };

    HRESULT InitConstantBuffer();
    void UpdateConstantBuffer(const DirectX::XMMATRIX& wvp);

public:
    Viewer(ID3D11Device* dev, ID3D11DeviceContext* ctx)
        : device(dev), context(ctx) {}

    HRESULT LoadShaders(const wchar_t* vsFile, const wchar_t* psFile);

    HRESULT CreateBuffers(const std::vector<Vector3D>& vertices, const std::vector<int>& indices);

    void Render(const DirectX::XMMATRIX& worldViewProj);

    void Cleanup();

    ~Viewer() { Cleanup(); }
};

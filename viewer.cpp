#include "viewer.h"
#include <d3dcompiler.h>
#include <fstream>
#include <cassert>

// Helper: Load compiled shader bytecode from file
static HRESULT LoadShaderFile(const wchar_t* filename, std::vector<char>& outData) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return E_FAIL;

    std::streamsize size = file.tellg();
    if (size <= 0) return E_FAIL;

    file.seekg(0, std::ios::beg);
    outData.resize(static_cast<size_t>(size));
    if (!file.read(outData.data(), size)) return E_FAIL;

    return S_OK;
}

HRESULT Viewer::InitConstantBuffer() {
    if (constantBuffer) {
        constantBuffer->Release();
        constantBuffer = nullptr;
    }

    D3D11_BUFFER_DESC desc{};
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.ByteWidth = sizeof(ConstantBuffer);
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

    return device->CreateBuffer(&desc, nullptr, &constantBuffer);
}

void Viewer::UpdateConstantBuffer(const DirectX::XMMATRIX& wvp) {
    ConstantBuffer cbData;
    cbData.worldViewProj = DirectX::XMMatrixTranspose(wvp); // Transpose for HLSL

    context->UpdateSubresource(constantBuffer, 0, nullptr, &cbData, 0, 0);

    context->VSSetConstantBuffers(0, 1, &constantBuffer);
}

HRESULT Viewer::LoadShaders(const wchar_t* vsFile, const wchar_t* psFile) {
    HRESULT hr = S_OK;
    std::vector<char> vsData, psData;

    hr = LoadShaderFile(vsFile, vsData);
    if (FAILED(hr)) return hr;

    hr = LoadShaderFile(psFile, psData);
    if (FAILED(hr)) return hr;

    hr = device->CreateVertexShader(vsData.data(), vsData.size(), nullptr, &vertexShader);
    if (FAILED(hr)) return hr;

    hr = device->CreatePixelShader(psData.data(), psData.size(), nullptr, &pixelShader);
    if (FAILED(hr)) return hr;

    // Define input layout matching the vertex shader input (position and normal)
    D3D11_INPUT_ELEMENT_DESC layoutDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,                             D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, sizeof(float) * 3,             D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = device->CreateInputLayout(layoutDesc, 2, vsData.data(), vsData.size(), &inputLayout);
    if (FAILED(hr)) return hr;

    hr = InitConstantBuffer();
    return hr;
}

struct Vertex {
    float x, y, z;
    float nx, ny, nz;
};

HRESULT Viewer::CreateBuffers(const std::vector<Vector3D>& vertices, const std::vector<int>& indices) {
    Cleanup(); // Release older buffers

    HRESULT hr = S_OK;

    // Prepare vertex data with positions and dummy normals (or compute real normals as needed)
    std::vector<Vertex> verts(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        verts[i].x = static_cast<float>(vertices[i].x);
        verts[i].y = static_cast<float>(vertices[i].y);
        verts[i].z = static_cast<float>(vertices[i].z);

        // For simplicity, set normal upwards
        verts[i].nx = 0.0f; verts[i].ny = 1.0f; verts[i].nz = 0.0f;
    }

    // Create Vertex Buffer
    D3D11_BUFFER_DESC vbDesc{};
    vbDesc.Usage = D3D11_USAGE_DEFAULT;
    vbDesc.ByteWidth = sizeof(Vertex) * (UINT)verts.size();
    vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA vbData{};
    vbData.pSysMem = verts.data();

    hr = device->CreateBuffer(&vbDesc, &vbData, &vertexBuffer);
    if (FAILED(hr)) return hr;

    // Create Index Buffer
    indexCount = (UINT)indices.size();

    D3D11_BUFFER_DESC ibDesc{};
    ibDesc.Usage = D3D11_USAGE_DEFAULT;
    ibDesc.ByteWidth = sizeof(int) * indexCount;
    ibDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;

    D3D11_SUBRESOURCE_DATA ibData{};
    ibData.pSysMem = indices.data();

    hr = device->CreateBuffer(&ibDesc, &ibData, &indexBuffer);
    if (FAILED(hr)) return hr;

    return hr;
}

void Viewer::Render(const DirectX::XMMATRIX& worldViewProj) {
    UpdateConstantBuffer(worldViewProj);

    UINT stride = sizeof(Vertex);
    UINT offset = 0;

    context->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
    context->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
    context->IASetInputLayout(inputLayout);

    context->VSSetShader(vertexShader, nullptr, 0);
    context->PSSetShader(pixelShader, nullptr, 0);

    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    context->DrawIndexed(indexCount, 0, 0);
}

void Viewer::Cleanup() {
    if (vertexBuffer) { vertexBuffer->Release(); vertexBuffer = nullptr; }
    if (indexBuffer) { indexBuffer->Release(); indexBuffer = nullptr; }
    if (constantBuffer) { constantBuffer->Release(); constantBuffer = nullptr; }
    if (inputLayout) { inputLayout->Release(); inputLayout = nullptr; }
    if (vertexShader) { vertexShader->Release(); vertexShader = nullptr; }
    if (pixelShader) { pixelShader->Release(); pixelShader = nullptr; }
}

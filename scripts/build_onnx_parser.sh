#!/usr/bin/env bash
# Build cust_onnx_parsers.so (includes BEVPoolV3 ONNX parser) for ATC.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASCEND="${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
PB="/usr/local/Ascend/mxVision-5.0.RC3/opensource"
BUILD="${ROOT}/build_onnx_parser"
OUT="${BUILD}/cust_onnx_parsers.so"

export PATH="${PB}/bin:${PATH}"
export LD_LIBRARY_PATH="${PB}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${BUILD}/autogen/proto/onnx"
if [[ ! -f "${BUILD}/autogen/proto/onnx/ge_onnx.pb.h" ]]; then
  protoc -I"${ASCEND}/include/proto" \
    --cpp_out="${BUILD}/autogen/proto/onnx" \
    "${ASCEND}/include/proto/ge_onnx.proto"
  mv "${BUILD}/autogen/proto/onnx/ge_onnx.pb."* "${BUILD}/autogen/proto/onnx/" 2>/dev/null || true
fi

g++ -shared -fPIC -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=ascend_private \
  -Wno-deprecated-declarations \
  -I"${ASCEND}/include" \
  -I"${PB}/include" \
  -I"${BUILD}/autogen" \
  -I"${ROOT}/include" \
  "${ROOT}/onnx_plugin/onnx_bev_pool_v3.cpp" \
  "${ROOT}/onnx_plugin/onnx_multi_scale_deformable_attn.cpp" \
  "${ROOT}/onnx_plugin/onnx_roi_align_rotated.cpp" \
  "${BUILD}/autogen/proto/onnx/ge_onnx.pb.cc" \
  -o "${OUT}" \
  -L"${ASCEND}/lib64" -L"${PB}/lib" \
  -lgraph -lregister -lge_common_base -lascendalog -lmindxsdk_protoc \
  -Wl,-rpath,"${ASCEND}/lib64:${PB}/lib"

mkdir -p "${ROOT}/mx_driving/packages/vendors/customize/framework/onnx"
cp "${OUT}" "${ROOT}/mx_driving/packages/vendors/customize/framework/onnx/"
echo "Built ${OUT}"
echo "Install: cd mx_driving && bash ../scripts/install_kernel.sh --quiet"

/* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * ONNX parser plugin: map ONNX custom op BEVPoolV3 -> CANN BEVPoolV3.
 * Install to packages/vendors/customize/framework/onnx/ (see onnx_plugin/CMakeLists.txt).
 */
#include "graph/operator.h"
#include "register/register.h"
#include "proto/onnx/ge_onnx.pb.h"

using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static bool ReadIntAttr(const NodeProto *node, const char *name, int64_t &out)
{
    for (const auto &attr : node->attribute()) {
        if (attr.name() == name && attr.type() == ge::onnx::AttributeProto::INT) {
            out = attr.i();
            return true;
        }
    }
    return false;
}

static bool ReadBoolAttr(const NodeProto *node, const char *name, bool &out)
{
    for (const auto &attr : node->attribute()) {
        if (attr.name() == name && attr.type() == ge::onnx::AttributeProto::INT) {
            out = (attr.i() != 0);
            return true;
        }
    }
    return false;
}

Status ParseOnnxParamsBEVPoolV3(const Message *op_src, ge::Operator &op_dest)
{
    const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
    if (node == nullptr) {
        return FAILED;
    }

    int64_t b = 1;
    int64_t d = 1;
    int64_t h = 1;
    int64_t w = 1;
    int64_t c = 1;
    bool with_depth = true;
    (void)ReadIntAttr(node, "b", b);
    (void)ReadIntAttr(node, "d", d);
    (void)ReadIntAttr(node, "h", h);
    (void)ReadIntAttr(node, "w", w);
    (void)ReadIntAttr(node, "c", c);
    (void)ReadBoolAttr(node, "with_depth", with_depth);

    op_dest.SetAttr("b", static_cast<int>(b));
    op_dest.SetAttr("d", static_cast<int>(d));
    op_dest.SetAttr("h", static_cast<int>(h));
    op_dest.SetAttr("w", static_cast<int>(w));
    op_dest.SetAttr("c", static_cast<int>(c));
    op_dest.SetAttr("with_depth", with_depth);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BEVPoolV3")
    .FrameworkType(ONNX)
    .OriginOpType({
        ge::AscendString("ai.onnx::8::BEVPoolV3"),
        ge::AscendString("ai.onnx::9::BEVPoolV3"),
        ge::AscendString("ai.onnx::10::BEVPoolV3"),
        ge::AscendString("ai.onnx::11::BEVPoolV3"),
        ge::AscendString("ai.onnx::12::BEVPoolV3"),
        ge::AscendString("ai.onnx::13::BEVPoolV3"),
    })
    .ParseParamsFn(ParseOnnxParamsBEVPoolV3)
    .ImplyType(ImplyType::TVM);
} // namespace domi

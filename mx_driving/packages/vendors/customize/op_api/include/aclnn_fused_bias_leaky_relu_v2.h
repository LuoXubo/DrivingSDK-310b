
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FUSED_BIAS_LEAKY_RELU_V2_H_
#define ACLNN_FUSED_BIAS_LEAKY_RELU_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFusedBiasLeakyReluV2GetWorkspaceSize
 * parameters :
 * x : required
 * bias : required
 * negativeSlope : required
 * scale : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFusedBiasLeakyReluV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    double negativeSlope,
    double scale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFusedBiasLeakyReluV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFusedBiasLeakyReluV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

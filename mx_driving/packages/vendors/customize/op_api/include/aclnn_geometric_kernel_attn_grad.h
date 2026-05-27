
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GEOMETRIC_KERNEL_ATTN_GRAD_H_
#define ACLNN_GEOMETRIC_KERNEL_ATTN_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGeometricKernelAttnGradGetWorkspaceSize
 * parameters :
 * value : required
 * spatialShapes : required
 * levelStartIndex : required
 * samplingLocations : required
 * attnWeights : required
 * gradOutput : required
 * gradValueOut : required
 * gradAttnWeightsOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGeometricKernelAttnGradGetWorkspaceSize(
    const aclTensor *value,
    const aclTensor *spatialShapes,
    const aclTensor *levelStartIndex,
    const aclTensor *samplingLocations,
    const aclTensor *attnWeights,
    const aclTensor *gradOutput,
    const aclTensor *gradValueOut,
    const aclTensor *gradAttnWeightsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGeometricKernelAttnGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGeometricKernelAttnGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

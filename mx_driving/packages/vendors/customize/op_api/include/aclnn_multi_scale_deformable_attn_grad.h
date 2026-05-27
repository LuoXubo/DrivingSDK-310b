
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MULTI_SCALE_DEFORMABLE_ATTN_GRAD_H_
#define ACLNN_MULTI_SCALE_DEFORMABLE_ATTN_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMultiScaleDeformableAttnGradGetWorkspaceSize
 * parameters :
 * value : required
 * spatialShapes : required
 * levelStartIndex : required
 * samplingLoc : required
 * attnWeight : required
 * gradOutput : required
 * gradValueOut : required
 * gradSamplingLocOut : required
 * gradAttnWeightOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMultiScaleDeformableAttnGradGetWorkspaceSize(
    const aclTensor *value,
    const aclTensor *spatialShapes,
    const aclTensor *levelStartIndex,
    const aclTensor *samplingLoc,
    const aclTensor *attnWeight,
    const aclTensor *gradOutput,
    const aclTensor *gradValueOut,
    const aclTensor *gradSamplingLocOut,
    const aclTensor *gradAttnWeightOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMultiScaleDeformableAttnGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMultiScaleDeformableAttnGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

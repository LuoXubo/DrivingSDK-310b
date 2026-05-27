
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MULTI_SCALE_DEFORMABLE_ATTN_H_
#define ACLNN_MULTI_SCALE_DEFORMABLE_ATTN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMultiScaleDeformableAttnGetWorkspaceSize
 * parameters :
 * value : required
 * valueSpatialShapes : required
 * valueLevelStartIndex : required
 * samplingLocations : required
 * attentionWeights : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMultiScaleDeformableAttnGetWorkspaceSize(
    const aclTensor *value,
    const aclTensor *valueSpatialShapes,
    const aclTensor *valueLevelStartIndex,
    const aclTensor *samplingLocations,
    const aclTensor *attentionWeights,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMultiScaleDeformableAttn
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMultiScaleDeformableAttn(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

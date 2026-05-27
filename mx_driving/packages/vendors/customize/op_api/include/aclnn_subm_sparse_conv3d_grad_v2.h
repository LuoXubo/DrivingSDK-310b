
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SUBM_SPARSE_CONV3D_GRAD_V2_H_
#define ACLNN_SUBM_SPARSE_CONV3D_GRAD_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSubmSparseConv3dGradV2GetWorkspaceSize
 * parameters :
 * features : required
 * weight : required
 * gradOutFeaturesOptional : optional
 * indicesOffset : required
 * featuresGradOut : required
 * weightGradOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dGradV2GetWorkspaceSize(
    const aclTensor *features,
    const aclTensor *weight,
    const aclTensor *gradOutFeaturesOptional,
    const aclTensor *indicesOffset,
    const aclTensor *featuresGradOut,
    const aclTensor *weightGradOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSubmSparseConv3dGradV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dGradV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

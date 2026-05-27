
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPARSE_CONV3D_GRAD_H_
#define ACLNN_SPARSE_CONV3D_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSparseConv3dGradGetWorkspaceSize
 * parameters :
 * features : required
 * weight : required
 * gradOutFeaturesOptional : optional
 * formerSortedIndices : required
 * indicesOffset : required
 * startOffset : required
 * endOffset : required
 * featuresGradOut : required
 * weightGradOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseConv3dGradGetWorkspaceSize(
    const aclTensor *features,
    const aclTensor *weight,
    const aclTensor *gradOutFeaturesOptional,
    const aclTensor *formerSortedIndices,
    const aclTensor *indicesOffset,
    int64_t startOffset,
    int64_t endOffset,
    const aclTensor *featuresGradOut,
    const aclTensor *weightGradOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSparseConv3dGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseConv3dGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SUBM_SPARSE_CONV3D_WITH_KEY_H_
#define ACLNN_SUBM_SPARSE_CONV3D_WITH_KEY_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSubmSparseConv3dWithKeyGetWorkspaceSize
 * parameters :
 * outidxOffset : required
 * validIndices : required
 * gradOutFeatures : required
 * kernelSize : required
 * inChannel : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dWithKeyGetWorkspaceSize(
    const aclTensor *outidxOffset,
    const aclTensor *validIndices,
    const aclTensor *gradOutFeatures,
    const aclIntArray *kernelSize,
    int64_t inChannel,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSubmSparseConv3dWithKey
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dWithKey(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

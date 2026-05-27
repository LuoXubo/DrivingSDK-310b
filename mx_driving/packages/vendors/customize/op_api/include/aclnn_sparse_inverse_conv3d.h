
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPARSE_INVERSE_CONV3D_H_
#define ACLNN_SPARSE_INVERSE_CONV3D_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSparseInverseConv3dGetWorkspaceSize
 * parameters :
 * features : required
 * indices : required
 * uniqueIndicesOffset : required
 * sortedIdxToFormerIndices : required
 * kernelSize : required
 * inChannels : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseInverseConv3dGetWorkspaceSize(
    const aclTensor *features,
    const aclTensor *indices,
    const aclTensor *uniqueIndicesOffset,
    const aclTensor *sortedIdxToFormerIndices,
    const aclIntArray *kernelSize,
    int64_t inChannels,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSparseInverseConv3d
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseInverseConv3d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

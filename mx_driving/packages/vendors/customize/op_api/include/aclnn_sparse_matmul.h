
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPARSE_MATMUL_H_
#define ACLNN_SPARSE_MATMUL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSparseMatmulGetWorkspaceSize
 * parameters :
 * features : required
 * weight : required
 * uniqueIndicesOffset : required
 * formerSortedIndices : required
 * indices : required
 * sparseValueOut : required
 * sparseIndicesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseMatmulGetWorkspaceSize(
    const aclTensor *features,
    const aclTensor *weight,
    const aclTensor *uniqueIndicesOffset,
    const aclTensor *formerSortedIndices,
    const aclTensor *indices,
    const aclTensor *sparseValueOut,
    const aclTensor *sparseIndicesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSparseMatmul
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseMatmul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

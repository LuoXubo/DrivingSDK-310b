
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BATCH_MATMUL_VECTOR_H_
#define ACLNN_BATCH_MATMUL_VECTOR_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBatchMatmulVectorGetWorkspaceSize
 * parameters :
 * projectionMat : required
 * ptsExtend : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBatchMatmulVectorGetWorkspaceSize(
    const aclTensor *projectionMat,
    const aclTensor *ptsExtend,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBatchMatmulVector
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBatchMatmulVector(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

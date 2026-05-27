
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GEOMETRIC_KERNEL_ATTENTION_H_
#define ACLNN_GEOMETRIC_KERNEL_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGeometricKernelAttentionGetWorkspaceSize
 * parameters :
 * value : required
 * spatialShapes : required
 * levelStartIndex : required
 * samplingLocations : required
 * attentionWeights : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGeometricKernelAttentionGetWorkspaceSize(
    const aclTensor *value,
    const aclTensor *spatialShapes,
    const aclTensor *levelStartIndex,
    const aclTensor *samplingLocations,
    const aclTensor *attentionWeights,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGeometricKernelAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGeometricKernelAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

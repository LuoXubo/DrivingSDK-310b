
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_SCATTER_GRAD_H_
#define ACLNN_DYNAMIC_SCATTER_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicScatterGradGetWorkspaceSize
 * parameters :
 * gradVoxelFeats : required
 * numPointPerVoxel : required
 * argsortCoor : required
 * compareMask : required
 * reduceType : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicScatterGradGetWorkspaceSize(
    const aclTensor *gradVoxelFeats,
    const aclTensor *numPointPerVoxel,
    const aclTensor *argsortCoor,
    const aclTensor *compareMask,
    char *reduceType,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicScatterGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicScatterGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

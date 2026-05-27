
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_SCATTER_H_
#define ACLNN_DYNAMIC_SCATTER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicScatterGetWorkspaceSize
 * parameters :
 * feats : required
 * prefixSumPointPerVoxel : required
 * argsortCoor : required
 * reduceType : required
 * voxelFeatsOut : required
 * compareMaskOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicScatterGetWorkspaceSize(
    const aclTensor *feats,
    const aclTensor *prefixSumPointPerVoxel,
    const aclTensor *argsortCoor,
    char *reduceType,
    const aclTensor *voxelFeatsOut,
    const aclTensor *compareMaskOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicScatter
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicScatter(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

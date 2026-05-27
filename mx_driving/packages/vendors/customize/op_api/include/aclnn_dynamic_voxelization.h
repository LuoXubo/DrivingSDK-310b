
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_VOXELIZATION_H_
#define ACLNN_DYNAMIC_VOXELIZATION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicVoxelizationGetWorkspaceSize
 * parameters :
 * points : required
 * coorsMinX : required
 * coorsMinY : required
 * coorsMinZ : required
 * voxelX : required
 * voxelY : required
 * voxelZ : required
 * gridX : required
 * gridY : required
 * gridZ : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicVoxelizationGetWorkspaceSize(
    const aclTensor *points,
    double coorsMinX,
    double coorsMinY,
    double coorsMinZ,
    double voxelX,
    double voxelY,
    double voxelZ,
    int64_t gridX,
    int64_t gridY,
    int64_t gridZ,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicVoxelization
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicVoxelization(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

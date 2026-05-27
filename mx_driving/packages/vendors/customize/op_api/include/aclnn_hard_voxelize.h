
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HARD_VOXELIZE_H_
#define ACLNN_HARD_VOXELIZE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHardVoxelizeGetWorkspaceSize
 * parameters :
 * points : required
 * uniVoxels : required
 * argsortVoxelIdices : required
 * uniArgsortIdices : required
 * uniIndices : required
 * numVoxels : required
 * maxVoxels : required
 * maxPoints : required
 * numPoints : required
 * voxelsOut : required
 * numPointsPerVoxelOut : required
 * sortedUniVoxelsOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHardVoxelizeGetWorkspaceSize(
    const aclTensor *points,
    const aclTensor *uniVoxels,
    const aclTensor *argsortVoxelIdices,
    const aclTensor *uniArgsortIdices,
    const aclTensor *uniIndices,
    int64_t numVoxels,
    int64_t maxVoxels,
    int64_t maxPoints,
    int64_t numPoints,
    const aclTensor *voxelsOut,
    const aclTensor *numPointsPerVoxelOut,
    const aclTensor *sortedUniVoxelsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHardVoxelize
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHardVoxelize(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

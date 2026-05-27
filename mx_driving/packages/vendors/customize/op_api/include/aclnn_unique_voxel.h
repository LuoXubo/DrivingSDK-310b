
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_UNIQUE_VOXEL_H_
#define ACLNN_UNIQUE_VOXEL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnUniqueVoxelGetWorkspaceSize
 * parameters :
 * voxels : required
 * indices : required
 * argsortIndices : required
 * uniVoxelsOut : required
 * uniIndicesOut : required
 * uniArgsortIndicesOut : required
 * voxelNumOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnUniqueVoxelGetWorkspaceSize(
    const aclTensor *voxels,
    const aclTensor *indices,
    const aclTensor *argsortIndices,
    const aclTensor *uniVoxelsOut,
    const aclTensor *uniIndicesOut,
    const aclTensor *uniArgsortIndicesOut,
    const aclTensor *voxelNumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnUniqueVoxel
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnUniqueVoxel(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

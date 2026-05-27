
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_VOXEL_TO_POINT_H_
#define ACLNN_VOXEL_TO_POINT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnVoxelToPointGetWorkspaceSize
 * parameters :
 * voxels : required
 * voxelSizes : required
 * coorRanges : required
 * layout : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVoxelToPointGetWorkspaceSize(
    const aclTensor *voxels,
    const aclFloatArray *voxelSizes,
    const aclFloatArray *coorRanges,
    char *layout,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnVoxelToPoint
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVoxelToPoint(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

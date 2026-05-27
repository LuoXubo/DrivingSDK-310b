
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_POINT_TO_VOXEL_H_
#define ACLNN_POINT_TO_VOXEL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnPointToVoxelGetWorkspaceSize
 * parameters :
 * points : required
 * voxelSizes : required
 * coorRanges : required
 * layout : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPointToVoxelGetWorkspaceSize(
    const aclTensor *points,
    const aclFloatArray *voxelSizes,
    const aclFloatArray *coorRanges,
    char *layout,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnPointToVoxel
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPointToVoxel(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

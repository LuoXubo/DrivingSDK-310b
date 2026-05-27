
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROIAWARE_POOL3D_H_
#define ACLNN_ROIAWARE_POOL3D_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRoiawarePool3dGetWorkspaceSize
 * parameters :
 * rois : required
 * pts : required
 * ptsFeature : required
 * mode : required
 * maxPtsEachVoxel : required
 * outx : required
 * outy : required
 * outz : required
 * argmaxOut : required
 * ptsIdxOfVoxelsOut : required
 * pooledFeaturesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiawarePool3dGetWorkspaceSize(
    const aclTensor *rois,
    const aclTensor *pts,
    const aclTensor *ptsFeature,
    int64_t mode,
    int64_t maxPtsEachVoxel,
    int64_t outx,
    int64_t outy,
    int64_t outz,
    const aclTensor *argmaxOut,
    const aclTensor *ptsIdxOfVoxelsOut,
    const aclTensor *pooledFeaturesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRoiawarePool3d
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiawarePool3d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

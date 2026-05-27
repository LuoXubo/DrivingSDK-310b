
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROIPOINT_POOL3D_FORWARD_H_
#define ACLNN_ROIPOINT_POOL3D_FORWARD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRoipointPool3dForwardGetWorkspaceSize
 * parameters :
 * points : required
 * pointFeatures : required
 * boxes3d : required
 * numSampledPoints : optional
 * pooledFeaturesOut : required
 * pooledEmptyFlagOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoipointPool3dForwardGetWorkspaceSize(
    const aclTensor *points,
    const aclTensor *pointFeatures,
    const aclTensor *boxes3d,
    int64_t numSampledPoints,
    const aclTensor *pooledFeaturesOut,
    const aclTensor *pooledEmptyFlagOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRoipointPool3dForward
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoipointPool3dForward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GAUSSIAN_H_
#define ACLNN_GAUSSIAN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGaussianGetWorkspaceSize
 * parameters :
 * gtBoxes : required
 * featureMapStride : required
 * gaussianOverlap : required
 * minRadius : required
 * numMaxObjs : required
 * voxelSizeX : required
 * voxelSizeY : required
 * pcRangeX : required
 * pcRangeY : required
 * featureMapSizeX : required
 * featureMapSizeY : required
 * normBbox : required
 * flipAngle : required
 * centerIntOut : required
 * radiusOut : required
 * maskOut : required
 * indOut : required
 * retBoxesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGaussianGetWorkspaceSize(
    const aclTensor *gtBoxes,
    int64_t featureMapStride,
    double gaussianOverlap,
    int64_t minRadius,
    int64_t numMaxObjs,
    double voxelSizeX,
    double voxelSizeY,
    double pcRangeX,
    double pcRangeY,
    int64_t featureMapSizeX,
    int64_t featureMapSizeY,
    bool normBbox,
    bool flipAngle,
    const aclTensor *centerIntOut,
    const aclTensor *radiusOut,
    const aclTensor *maskOut,
    const aclTensor *indOut,
    const aclTensor *retBoxesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGaussian
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGaussian(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

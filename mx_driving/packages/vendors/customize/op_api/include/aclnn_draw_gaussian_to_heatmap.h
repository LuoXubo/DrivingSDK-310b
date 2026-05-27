
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DRAW_GAUSSIAN_TO_HEATMAP_H_
#define ACLNN_DRAW_GAUSSIAN_TO_HEATMAP_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDrawGaussianToHeatmapGetWorkspaceSize
 * parameters :
 * mask : required
 * curClassId : required
 * centerInt : required
 * radius : required
 * numClasses : required
 * featureMapSizeX : required
 * featureMapSizeY : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDrawGaussianToHeatmapGetWorkspaceSize(
    const aclTensor *mask,
    const aclTensor *curClassId,
    const aclTensor *centerInt,
    const aclTensor *radius,
    int64_t numClasses,
    int64_t featureMapSizeX,
    int64_t featureMapSizeY,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDrawGaussianToHeatmap
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDrawGaussianToHeatmap(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

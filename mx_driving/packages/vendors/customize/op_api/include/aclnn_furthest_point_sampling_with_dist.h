
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FURTHEST_POINT_SAMPLING_WITH_DIST_H_
#define ACLNN_FURTHEST_POINT_SAMPLING_WITH_DIST_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFurthestPointSamplingWithDistGetWorkspaceSize
 * parameters :
 * pointsDist : required
 * nearestTemp : required
 * numPoints : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFurthestPointSamplingWithDistGetWorkspaceSize(
    const aclTensor *pointsDist,
    const aclTensor *nearestTemp,
    int64_t numPoints,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFurthestPointSamplingWithDist
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFurthestPointSamplingWithDist(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

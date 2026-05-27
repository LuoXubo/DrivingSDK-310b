
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_PIXEL_GROUP_H_
#define ACLNN_PIXEL_GROUP_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnPixelGroupGetWorkspaceSize
 * parameters :
 * score : required
 * mask : required
 * embedding : required
 * kernelLabel : required
 * kernelContour : required
 * kernelRegionNum : required
 * distanceThreshold : required
 * pointVectorOut : required
 * labelUpdatedOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPixelGroupGetWorkspaceSize(
    const aclTensor *score,
    const aclTensor *mask,
    const aclTensor *embedding,
    const aclTensor *kernelLabel,
    const aclTensor *kernelContour,
    int64_t kernelRegionNum,
    double distanceThreshold,
    const aclTensor *pointVectorOut,
    const aclTensor *labelUpdatedOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnPixelGroup
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPixelGroup(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

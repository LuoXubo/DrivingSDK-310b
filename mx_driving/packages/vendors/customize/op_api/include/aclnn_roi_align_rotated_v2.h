
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROI_ALIGN_ROTATED_V2_H_
#define ACLNN_ROI_ALIGN_ROTATED_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRoiAlignRotatedV2GetWorkspaceSize
 * parameters :
 * input : required
 * rois : required
 * spatialScale : required
 * samplingRatio : required
 * pooledHeight : required
 * pooledWidth : required
 * aligned : required
 * clockwise : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiAlignRotatedV2GetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *rois,
    double spatialScale,
    int64_t samplingRatio,
    int64_t pooledHeight,
    int64_t pooledWidth,
    bool aligned,
    bool clockwise,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRoiAlignRotatedV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiAlignRotatedV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

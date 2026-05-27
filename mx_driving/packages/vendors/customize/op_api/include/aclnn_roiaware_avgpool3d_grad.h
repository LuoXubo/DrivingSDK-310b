
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROIAWARE_AVGPOOL3D_GRAD_H_
#define ACLNN_ROIAWARE_AVGPOOL3D_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRoiawareAvgpool3dGradGetWorkspaceSize
 * parameters :
 * ptsIdxOfVoxels : required
 * gradOut : required
 * boxesNum : required
 * outX : required
 * outY : required
 * outZ : required
 * channels : required
 * npoints : required
 * maxPtsPerVoxel : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiawareAvgpool3dGradGetWorkspaceSize(
    const aclTensor *ptsIdxOfVoxels,
    const aclTensor *gradOut,
    int64_t boxesNum,
    int64_t outX,
    int64_t outY,
    int64_t outZ,
    int64_t channels,
    int64_t npoints,
    int64_t maxPtsPerVoxel,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRoiawareAvgpool3dGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiawareAvgpool3dGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

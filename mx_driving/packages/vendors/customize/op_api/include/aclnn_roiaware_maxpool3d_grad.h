
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROIAWARE_MAXPOOL3D_GRAD_H_
#define ACLNN_ROIAWARE_MAXPOOL3D_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRoiawareMaxpool3dGradGetWorkspaceSize
 * parameters :
 * argmax : required
 * gradOut : required
 * boxesNum : required
 * outX : required
 * outY : required
 * outZ : required
 * channels : required
 * npoints : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiawareMaxpool3dGradGetWorkspaceSize(
    const aclTensor *argmax,
    const aclTensor *gradOut,
    int64_t boxesNum,
    int64_t outX,
    int64_t outY,
    int64_t outZ,
    int64_t channels,
    int64_t npoints,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRoiawareMaxpool3dGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiawareMaxpool3dGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

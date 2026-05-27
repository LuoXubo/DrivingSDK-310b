
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GRID_SAMPLER3D_GRAD_V1_H_
#define ACLNN_GRID_SAMPLER3D_GRAD_V1_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGridSampler3dGradV1GetWorkspaceSize
 * parameters :
 * grad : required
 * x : required
 * grid : required
 * interpolationMode : required
 * paddingMode : required
 * alignCorners : required
 * dxOut : required
 * dgridOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGridSampler3dGradV1GetWorkspaceSize(
    const aclTensor *grad,
    const aclTensor *x,
    const aclTensor *grid,
    int64_t interpolationMode,
    int64_t paddingMode,
    bool alignCorners,
    const aclTensor *dxOut,
    const aclTensor *dgridOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGridSampler3dGradV1
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGridSampler3dGradV1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

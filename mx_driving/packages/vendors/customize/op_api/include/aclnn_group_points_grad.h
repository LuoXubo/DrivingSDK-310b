
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GROUP_POINTS_GRAD_H_
#define ACLNN_GROUP_POINTS_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGroupPointsGradGetWorkspaceSize
 * parameters :
 * gradOut : required
 * indices : required
 * b : required
 * c : required
 * n : required
 * npoints : required
 * nsample : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupPointsGradGetWorkspaceSize(
    const aclTensor *gradOut,
    const aclTensor *indices,
    int64_t b,
    int64_t c,
    int64_t n,
    int64_t npoints,
    int64_t nsample,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGroupPointsGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupPointsGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

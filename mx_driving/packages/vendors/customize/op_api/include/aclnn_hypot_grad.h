
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HYPOT_GRAD_H_
#define ACLNN_HYPOT_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHypotGradGetWorkspaceSize
 * parameters :
 * x : required
 * y : required
 * z : required
 * zGrad : required
 * xGradOut : required
 * yGradOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHypotGradGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *z,
    const aclTensor *zGrad,
    const aclTensor *xGradOut,
    const aclTensor *yGradOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHypotGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHypotGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BEVPOOL_GRAD_H_
#define ACLNN_BEVPOOL_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBEVPoolGradGetWorkspaceSize
 * parameters :
 * gradOut : required
 * geomFeat : required
 * intervalLengths : required
 * intervalStarts : required
 * b : required
 * d : required
 * h : required
 * w : required
 * c : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBEVPoolGradGetWorkspaceSize(
    const aclTensor *gradOut,
    const aclTensor *geomFeat,
    const aclTensor *intervalLengths,
    const aclTensor *intervalStarts,
    int64_t b,
    int64_t d,
    int64_t h,
    int64_t w,
    int64_t c,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBEVPoolGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBEVPoolGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif


/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCATTER_ADD_V1_H_
#define ACLNN_SCATTER_ADD_V1_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScatterAddV1GetWorkspaceSize
 * parameters :
 * src : required
 * indices : required
 * var : required
 * dim : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterAddV1GetWorkspaceSize(
    const aclTensor *src,
    const aclTensor *indices,
    const aclTensor *var,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterAddV1
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterAddV1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

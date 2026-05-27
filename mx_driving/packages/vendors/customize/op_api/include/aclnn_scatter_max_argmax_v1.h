
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCATTER_MAX_ARGMAX_V1_H_
#define ACLNN_SCATTER_MAX_ARGMAX_V1_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScatterMaxArgmaxV1GetWorkspaceSize
 * parameters :
 * src : required
 * index : required
 * resOut : required
 * argmaxOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterMaxArgmaxV1GetWorkspaceSize(
    const aclTensor *src,
    const aclTensor *index,
    const aclTensor *resOut,
    const aclTensor *argmaxOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterMaxArgmaxV1
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterMaxArgmaxV1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
